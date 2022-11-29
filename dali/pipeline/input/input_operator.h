// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_INPUT_INPUT_OPERATOR_H_
#define DALI_PIPELINE_INPUT_INPUT_OPERATOR_H_

#include <list>
#include <memory>
#include "dali/core/common.h"
#include "dali/core/cuda_event.h"
#include "dali/pipeline/input/caching_list.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace detail {

struct CudaEventWrapper : CUDAEvent {
  CudaEventWrapper() : CUDAEvent(CUDAEvent::Create()) {}
};

}  // namespace detail

template<typename Backend>
class InputOperator : public Operator<Backend> {
 public:
  explicit InputOperator(const OpSpec &spec) :
          Operator<Backend>(spec),
          device_id_(spec.GetArgument<int>("device_id")) {
    if (std::is_same<Backend, GPUBackend>::value) {
      internal_copy_stream_ = CUDAStreamPool::instance().Get(device_id_);
      internal_copy_order_ = internal_copy_stream_;
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(InputOperator);

 protected:
  using Operator<Backend>::spec_;
  using uptr_tl_type = std::unique_ptr<TensorList<Backend>>;
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;


  // pass cuda_event by pointer to allow default, nullptr value, with the
  // reference it is not that easy
  template<typename DataType>
  void RecycleBuffer(DataType &data,
                     std::list<uptr_cuda_event_type> *cuda_event = nullptr,
                     std::list<uptr_cuda_event_type> *copy_to_gpu = nullptr) {
    // No need to synchronize on copy_to_gpu - it was already synchronized before
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    tl_data_.Recycle(data);
    if (copy_to_gpu) {
      copy_to_storage_events_.Recycle(*copy_to_gpu);
    }
  }


  template<typename SrcBackend>
  inline std::enable_if_t<!std::is_same<SrcBackend, Backend>::value>
  ShareUserData(const TensorList<SrcBackend> &t, AccessOrder /* order = {}*/,
                bool /* use_copy_kernel */) {
    DALI_FAIL(make_string("no_copy is supported only for the same data source device type "
                          "as operator. Received: ",
                          std::is_same<SrcBackend, CPUBackend>::value ? "CPU" : "GPU",
                          " input for ",
                          std::is_same<Backend, CPUBackend>::value ? "CPU" : "GPU",
                          " operator."));
  }


  template<typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder /* order = {}*/,
                bool /*use_copy_kernel = false*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({false, true});
    auto tl_elm = GetEmptyOutputBatch();
    // set pinned if needed
    if (batch.is_pinned() != tl_elm.front()->is_pinned()) {
      tl_elm.front()->Reset();
      tl_elm.front()->set_pinned(batch.is_pinned());
    }
    tl_elm.front()->ShareData(const_cast<TensorList<CPUBackend> &>(batch));
    tl_data_.PushBack(tl_elm);
  }


  /**
   * @brief Attempts to share data from tensor vector to tensor list without
   *        an additional copy if the batch is contiguous.
   *        In case of scattered samples, the data is copied to a contiguous
   *        buffer.
   * @remarks Mixing contiguous and non-contiguous inputs in subsequents calls
   *        is not supported and could lead to data corruption.
   * @param batch source data
   * @param order CUDA stream use to schedule the copy (or host order to make the copy
   *              host-syncrhonous)
   * @param use_copy_kernel If true, a copy kernel will be used to make a
   *        contiguous buffer instead of cudaMemcpyAsync.
   */
  template<typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder order = {},
                bool use_copy_kernel = false) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = GetEmptyOutputBatch();
    bool copied_shared_data = false;

    if (batch.IsContiguous()) {
      auto batch_owner = unsafe_sample_owner(const_cast<TensorList<SrcBackend> &>(batch), 0);
      tl_elm.front()->ShareData(batch_owner, batch.nbytes(), batch.is_pinned(), batch.shape(),
                                batch.type(), batch.device_id(), batch.order());
      zero_copy_noncontiguous_gpu_input_ = true;
    } else {
      // it is not contiguous so we need to copy
      tl_elm.front()->Copy(batch, order, use_copy_kernel);
      std::list<uptr_cuda_event_type> copy_to_storage_event;
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
      CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), order.stream()));
      copy_to_storage_events_.PushBack(copy_to_storage_event);

      if (zero_copy_noncontiguous_gpu_input_) {
        DALI_WARN("ExternalSource operator should not mix contiguous and noncontiguous inputs. "
                  "In such a case the internal memory used to gather data in a contiguous chunk "
                  "of memory would be trashed.");
      }
      copied_shared_data = true;
    }
    state_.push_back({copied_shared_data, true});
    tl_data_.PushBack(tl_elm);
  }


  template<typename SrcBackend, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, CPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch,
               AccessOrder order, bool /* sync */, bool /* use_copy_kernel */) {
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch();
    }
    // set pinned if needed
    tl_elm.front()->set_order(AccessOrder::host());
    if (batch.is_pinned() != tl_elm.front()->is_pinned()) {
      tl_elm.front()->Reset();
      tl_elm.front()->set_pinned(batch.is_pinned());
    }
    AccessOrder copy_order =
            std::is_same<SrcBackend, CPUBackend>::value
            ? AccessOrder::host()  // do not use a device order for a host to host copy
            : order;
    tl_elm.front()->Copy(batch, copy_order);
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.PushBack(tl_elm);
      state_.push_back({false, false});
    }
  }


  template<typename SrcBackend, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, GPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch,
               AccessOrder order, bool sync, bool use_copy_kernel) {
    std::list<uptr_cuda_event_type> copy_to_storage_event;
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch();
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
    }
    // If we got a host order we most probably got it via FeedPipeline and we are trying to pass the
    // data from CPU to GPU. As we keep the order in tl_data_ as internal_copy_stream_, we will use
    // an actual stream for running and synchronizing with the copy. Note that the Copy can be truly
    // asynchronous if it comes from pinned memory or happens on a device with integrated memory
    // (like Xavier) where CPU and GPU share the same memory.
    if (!order.is_device()) {
      order = tl_elm.front()->order();
    }
    tl_elm.front()->Copy(batch, order, use_copy_kernel);
    CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), order.stream()));
    if (sync) {
      CUDA_CALL(cudaEventSynchronize(*copy_to_storage_event.front()));
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.PushBack(tl_elm);
      copy_to_storage_events_.PushBack(copy_to_storage_event);
      state_.push_back({false, false});
    }
  }


  CachingList<uptr_tl_type> &GetOutputDataQueue() {
    return tl_data_;
  }


  bool HasData() const {
    return !state_.empty();
  }


  std::mutex busy_m_;

  struct InputSourceState {
    /**
     * True if the data that was shared as no_copy required copy regardless.
     * Happens for non-contiguous TensorList with GPU memory.
     */
    bool copied_shared_data = false;
    /**
     * @brief Actual value of no_copy option used in this call. Always false for CopyUserData(...)
     * and always true for ShareUserData(...)
     */
    bool no_copy = false;
  };

  std::list<InputSourceState> state_;

 private:
  /**
   * @brief Get the empty output batch from tl_data_, first assigning the correct order to it.
   * @warning User is responsible for holding busy_m_ mutex when calling this function.
   */
  std::list<uptr_tl_type> GetEmptyOutputBatch() {
    auto result = tl_data_.GetEmpty();
    result.front()->set_order(internal_copy_order_);
    return result;
  }


  CachingList<uptr_tl_type> tl_data_;
  CachingList<uptr_cuda_event_type> copy_to_storage_events_;


  /**
   * Indicates that user provide noncontiguous GPU input with zero copy option so DALI needs
   * to create an internal copy, it is used to raise a warning when the user mixes contiguous and
   * noncontiguous GPU inputs with zero copy what trashed GPU allocated memory
   */
  bool zero_copy_noncontiguous_gpu_input_ = false;

  int device_id_;
  CUDAStreamLease internal_copy_stream_ = {};
  AccessOrder internal_copy_order_ = AccessOrder::host();
};

}  // namespace dali

#endif  // DALI_PIPELINE_INPUT_INPUT_OPERATOR_H_
