// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_

#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <mutex>
#include <condition_variable>
#include <utility>

#include "dali/core/cuda_event.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace detail {

struct CudaEventWrapper : CUDAEvent {
  CudaEventWrapper() : CUDAEvent(CUDAEvent::Create()) {}
};

/**
 * CachingList differs from std::List by the ability to recycle empty elements. When allocating memory
 * is expensive it is better to store already allocated but no longer needed element in the list of the free
 * elements, than to free the memory and allocate it again later. CachingList supports the following operations:
 * - GetEmpty moves an empty element of type T, either allocate it or use one from the free list
 * - PopFront moves the element from the front and removes it from the full list, the behavior
 * is undefined when the list is empty
 * - Recycle moves passed element to the free list
 * - PushBack moves element to the full list
 * - IsEmpty checks if the full list is empty
 * All functions operate on one element list as transferring elements between list is a very low cost
 * operation, which doesn't involve any memory allocation, while adding an element to the list requires
 * allocation of the memory for the storage in the list.
 */
template <typename T>
class CachingList {
 public:
  bool IsEmpty() const {
    return full_data_.empty();
  }

  T &PeekFront() {
    return full_data_.front();
  }

  std::list<T> PopFront() {
    std::list<T> tmp;
    tmp.splice(tmp.begin(), full_data_, full_data_.begin());
    return tmp;
  }

  void Recycle(std::list<T> &elm) {
    empty_data_.splice(empty_data_.end(), elm, elm.begin());
  }

  std::list<T> GetEmpty() {
    std::list<T> tmp;
    if (empty_data_.empty()) {
      tmp.emplace_back(std::make_unique<typename T::element_type>());
    } else {
      tmp.splice(tmp.begin(), empty_data_, empty_data_.begin());
    }
    return tmp;
  }

  void PushBack(std::list<T> &elm) {
    full_data_.splice(full_data_.end(), elm, elm.begin());
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;
};

}  // namespace detail

/**
 * @brief Provides in-graph access to data fed in from outside of dali.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 * Please note, that it is not allowed to call this concurrently as it
 * may mix the order of inputted data.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend> {
  using uptr_tl_type = std::unique_ptr<TensorList<Backend>>;
  using uptr_tv_type = std::unique_ptr<TensorVector<Backend>>;
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;

 public:
  inline explicit ExternalSource(const OpSpec &spec) :
    Operator<Backend>(spec),
    blocking_(spec.GetArgument<bool>("blocking")),
    no_copy_(spec.GetArgument<bool>("no_copy")),
    device_id_(spec.GetArgument<int>("device_id")),
    sync_worker_(device_id_, false) {
    output_name_ = spec.Output(0);
    sync_worker_.WaitForInit();
  }

  inline ~ExternalSource() {
    try {
      sync_worker_.ForceStop();
      sync_worker_.Shutdown();
    } catch (const std::exception &) {
      // Something went terribly wrong while releasing resources. We'd better die right now.
      std::terminate();
    }
  }

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  template<typename SrcBackend>
  inline void SetDataSource(const TensorList<SrcBackend> &tl, cudaStream_t stream = 0,
                            bool sync = false) {
    DeviceGuard g(device_id_);
    SetDataSourceHelper(tl, stream, sync);
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  template<typename SrcBackend>
  inline void SetDataSource(const vector<Tensor<SrcBackend>> &vect_of_tensors,
                            cudaStream_t stream = 0, bool sync = false) {
    DeviceGuard g(device_id_);
    TensorVector<SrcBackend> tv(vect_of_tensors.size());
    for (size_t i = 0; i < tv.size(); ++i) {
      tv[i].ShareData(const_cast<Tensor<SrcBackend>*>(&vect_of_tensors[i]));
    }
    SetDataSourceHelper(tv, stream, sync);
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  template<typename SrcBackend>
  inline void SetDataSource(const TensorVector<SrcBackend> &tv, cudaStream_t stream = 0,
                            bool sync = false) {
    DeviceGuard g(device_id_);
    SetDataSourceHelper(tv, stream, sync);
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    // for zero copy we don't want any shape inference to avoid any data allocation ahead
    if (no_copy_) {
      return false;
    }
    TensorListShape<> shape;
    output_desc.resize(1);
    {
      std::unique_lock<std::mutex> busy_lock(busy_m_);
      if (blocking_) {
        cv_.wait(busy_lock, [&data = state_] {return !data.empty(); });
      } else {
        if (state_.empty()) {
          DALI_FAIL("No data was provided to the ExternalSource. Make sure to feed it properly.");
        }
      }
      if (std::is_same<Backend, GPUBackend>::value) {
        output_desc[0].shape = tl_data_.PeekFront()->shape();
        output_desc[0].type = tl_data_.PeekFront()->type();
      } else {
        output_desc[0].shape = tv_data_.PeekFront()->shape();
        output_desc[0].type = tv_data_.PeekFront()->type();
      }
    }
    return true;
  }

  bool CanInferOutputs() const override {
    // when it passes through no shape inference is needed
    return !no_copy_;
  }

  /*
   * So that compiler wouldn't complain, that
   * "overloaded virtual function `dali::Operator<dali::CPUBackend>::RunImpl` is only partially
   * overridden in class `dali::[...]<dali::CPUBackend>`"
   */
  using Operator<Backend>::RunImpl;

  void RunImpl(workspace_t<Backend> &ws) override;

  void RecycleBufferHelper(std::list<uptr_tl_type> &data) {
    tl_data_.Recycle(data);
  }

  void RecycleBufferHelper(std::list<uptr_tv_type> &data) {
    tv_data_.Recycle(data);
  }

  // pass cuda_event by pointer to allow default, nullptr value, with the
  // reference it is not that easy
  template<typename DataType>
  void RecycleBuffer(DataType &data,
                     std::list<uptr_cuda_event_type> *cuda_event = nullptr,
                     std::list<uptr_cuda_event_type> *copy_to_gpu = nullptr) {
    if (cuda_event) {
      cudaEventSynchronize(*cuda_event->front());
    }
    // No need to synchronize on copy_to_gpu - it was already synchronized before
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    RecycleBufferHelper(data);
    if (cuda_event) {
      cuda_events_.Recycle(*cuda_event);
    }
    if (copy_to_gpu) {
      copy_to_storage_events_.Recycle(*copy_to_gpu);
    }
  }

  template<typename SrcBackend, template<typename> class SourceDataType>
  inline std::enable_if_t<!std::is_same<SrcBackend, Backend>::value>
  ShareUserData(const SourceDataType<SrcBackend> &t, cudaStream_t /*stream = 0*/) {
    DALI_FAIL("no_copy is supported only for the same data source device type as operator.");
  }

  template<typename SrcBackend, template<typename> class SourceDataType>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const SourceDataType<SrcBackend> &t, cudaStream_t /*stream = 0*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({});
    auto tv_elm = tv_data_.GetEmpty();
    tv_elm.front()->ShareData(const_cast<SourceDataType<CPUBackend>*>(&t));
    tv_data_.PushBack(tv_elm);
  }

  template<typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorVector<SrcBackend> &t, cudaStream_t stream = 0) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = tl_data_.GetEmpty();
    if (t.IsContiguous()) {
      t.ShareWith(const_cast<TensorList<Backend>*>(tl_elm.front().get()));
      zero_copy_noncontiguous_gpu_input_ = true;
      state_.push_back({});
    } else {
      // it is not contiguous so we need to copy
      tl_elm.front()->Copy(t, stream);

      std::list<uptr_cuda_event_type> copy_to_storage_event;
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
      cudaEventRecord(*copy_to_storage_event.front(), stream);
      copy_to_storage_events_.PushBack(copy_to_storage_event);

      if (zero_copy_noncontiguous_gpu_input_) {
        DALI_WARN("ExternalSource operator should not mix contiguous and noncontiguous inputs. "
                  "In such a case the internal memory used to gather data in a contiguous chunk "
                  "of memory would be trashed.");
      }
      state_.push_back({true});
    }
    tl_data_.PushBack(tl_elm);
  }

  template<typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
   ShareUserData(const TensorList<SrcBackend> &t, cudaStream_t /*stream = 0*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({});
    auto tl_elm = tl_data_.GetEmpty();
    tl_elm.front()->ShareData(const_cast<TensorList<Backend>*>(&t));
    tl_data_.PushBack(tl_elm);
    zero_copy_noncontiguous_gpu_input_ = true;
  }

  template<typename SrcBackend, template<typename> class SourceDataType, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, CPUBackend>::value>
  CopyUserData(const SourceDataType<SrcBackend> &batch, cudaStream_t /*stream*/, bool /*sync*/) {
    std::list<uptr_tv_type> tv_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tv_elm = tv_data_.GetEmpty();
    }
    // if it was not allocated already set_pinned to false
    if (!tv_elm.front()->size()) {
      tv_elm.front()->set_pinned(false);
    }
    // HostWorkspace doesn't have any stream
    cudaStream_t stream = 0;
    tv_elm.front()->Copy(batch, stream);
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tv_data_.PushBack(tv_elm);
      state_.push_back({});
    }
  }

  template<typename SrcBackend, template<typename> class SourceDataType, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, GPUBackend>::value>
  CopyUserData(const SourceDataType<SrcBackend> &batch, cudaStream_t stream, bool sync) {
    std::list<uptr_cuda_event_type> copy_to_storage_event;
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = tl_data_.GetEmpty();
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
    }
    tl_elm.front()->Copy(batch, stream);
    // record event for:
    // - GPU -> GPU
    // - pinned CPU -> GPU
    if (std::is_same<SrcBackend, GPUBackend>::value || batch.is_pinned()) {
      cudaEventRecord(*copy_to_storage_event.front(), stream);
    }
    // sync for pinned CPU -> GPU as well, because the user doesn't know when he can
    // reuse provided memory anyway
    if (sync || batch.is_pinned()) {
      CUDA_CALL(cudaEventSynchronize(*copy_to_storage_event.front()));
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.PushBack(tl_elm);
      copy_to_storage_events_.PushBack(copy_to_storage_event);
      state_.push_back({});
    }
  }

  template<typename SrcBackend, template<typename> class SourceDataType>
  inline void SetDataSourceHelper(const SourceDataType<SrcBackend> &batch, cudaStream_t stream = 0,
                                  bool sync = false) {
    bool is_gpu_src = std::is_same<SrcBackend, GPUBackend>::value;
    bool is_gpu_dst = std::is_same<Backend, GPUBackend>::value;
    if (is_gpu_src && !is_gpu_dst) {
      DALI_WARN("Warning: Loading GPU-originated data into CPU "
                "ExternalSource operator is discouraged and might be inefficient.");
    }
    DALI_ENFORCE(OperatorBase::batch_size_ == static_cast<int>(batch.ntensor()),
                 make_string("Data list provided to ExternalSource needs to have batch_size = ",
                             OperatorBase::batch_size_, " length, found ", batch.ntensor(),
                             " samples."));
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    std::list<uptr_tl_type> tl_elm;
    std::list<uptr_tl_type> tv_elm;
    if (no_copy_) {
      ShareUserData(batch, stream);
    } else {
      CopyUserData(batch, stream, sync);
    }
    cv_.notify_one();
  }

  string output_name_;
  detail::CachingList<uptr_tl_type> tl_data_;
  detail::CachingList<uptr_tv_type> tv_data_;
  detail::CachingList<uptr_cuda_event_type> cuda_events_, copy_to_storage_events_;
  struct RecycleFunctor;

  std::mutex busy_m_;
  std::condition_variable cv_;
  bool blocking_ = true;
  bool no_copy_ = false;
  int device_id_;

  /*
   * now it only indicates that there is data in the ExternalSource, in the future
   * a per sample metadata could be stored here
   */
  struct ExternalSourceState {
    bool copied_shared_data = false;
  };

  std::list<ExternalSourceState > state_;

  /*
   * indicates that user provide noncontiguous GPU input with zero copy option so DALI needs
   * to create an internal copy, it is used to raise a warning when the user mixes contiguous and
   * noncontiguous GPU inputs with zero copy what trashed GPU allocated memory
   */
  bool zero_copy_noncontiguous_gpu_input_ = false;

  WorkerThread sync_worker_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
