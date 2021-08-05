// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/cuda_event.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"
#include "dali/core/common.h"

namespace dali {

namespace detail {

struct CudaEventWrapper : CUDAEvent {
  CudaEventWrapper() : CUDAEvent(CUDAEvent::Create()) {}
};

/**
 * CachingList differs from std::List by the ability to recycle empty elements. When allocating
 * memory is expensive it is better to store already allocated but no longer needed element in the
 * list of the free elements, than to free the memory and allocate it again later. CachingList
 * supports the following operations:
 * - GetEmpty moves an empty element of type T, either allocate it or use one from the free list
 * - PopFront moves the element from the front and removes it from the full list, the behavior
 * is undefined when the list is empty
 * - Recycle moves passed element to the free list
 * - PushBack moves element to the full list
 * - IsEmpty checks if the full list is empty
 * All functions operate on one element list as transferring elements between list is a very low
 * cost operation, which doesn't involve any memory allocation, while adding an element to the list
 * requires allocation of the memory for the storage in the list.
 *
 * Additionally, CachingList has a Prophet feature. This is an unidirectional iterator,
 * that travels over the data (asynchronously w.r.t. current Front and Back). The Prophet
 * allows to peek a list element and maintains the order even when elements are Pushed
 * and Popped in/out.
 * Use PeekProphet() and AdvanceProphet() to control the prophet.
 * In case there's an illegal access to the list, std::out_of_range will be thrown.
 */
template <typename T>
class CachingList {
 public:
  CachingList() : prophet_(full_data_.end()) {}

  bool IsEmpty() const {
    return full_data_.empty();
  }

  const T &PeekFront() {
    return full_data_.front();
  }

  std::list<T> PopFront() {
    assert(!full_data_.empty());  // Can't pop from an empty list
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
    /*
     * When the prophet is dead and needs to be resurrected,
     * he shall be resurrected by the apprentice.
     * In the special scenario, when prophet is dead and the data list is empty
     * (hence the apprentice is dead too), the prophet will be resurrected
     * from scratch, by assigning him to the element that was just added to the data list.
     * Sic mundus creatus est.
     */
    if (resurrect_prophet_) {
      if (full_data_.size() == 1) {
        prophet_ = full_data_.begin();
      } else {
        prophet_ = std::next(apprentice_);
      }
      resurrect_prophet_ = false;
    }
  }

  const T &PeekProphet() {
    if (prophet_ == full_data_.end())
      throw std::out_of_range(
          "Attempted to peek element that doesn't exist. Add more elements to CachingList before "
          "calling PeekProphet. Even the prophet can't see outside the event horizon.");
    return *prophet_;
  }

  void AdvanceProphet() {
    if (prophet_ == full_data_.end())
      throw std::out_of_range(
          "Attempted to step over the last element in the list. This operation is forbidden. Add "
          "more elements to CachingList before calling AdvanceProphet.");
    apprentice_ = prophet_++;
    resurrect_prophet_ = prophet_ == full_data_.end();
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;

  /**
   * Prophet dies when he hits the end() iterator of the list with the data.
   * Prophet can be resurrected, iff there is a data record for him, i.e.
   * when user calls PushBack and therefore inserts the data at the end
   * of the CachingList
   */
  bool resurrect_prophet_ = true;

  /**
   * The apprentice follows the prophet and is always one step behind him.
   * Apprentice is used to resurrect the prophet, so that the prophet might
   * again point to the last actual element of the list.
   */
  typename std::list<T>::iterator prophet_, apprentice_;
};

template<typename Backend, template<typename>class BatchContainer>
auto get_batch_size(const BatchContainer<Backend>& container) {
  static_assert(is_batch_container<BatchContainer, Backend>::value,
      "Invalid container. Use TensorVector/TensorList and CPUBackend/GPUBackend/MixedBackend.");
  return container.ntensor();
}

}  // namespace detail


/**
 * @brief Option used to override the External Source no_copy parameter
 *
 * It allows to:
 *  * DEFAULT - leave the default (the no_copy parameter is used),
 *  * FORCE_COPY - always make a copy,
 *  * FORCE_NO_COPY - always share the data without copy.
 */
enum class ExtSrcNoCopyMode {
  DEFAULT,
  FORCE_COPY,
  FORCE_NO_COPY
};


/**
 * @brief Options that can be configured when setting data for the External Source
 */
struct ExtSrcSettingMode {
  /**
   * @brief If SetExternalInputHelper should be blocking - waits until provided data is copied
   *        to the internal buffer
   */
  bool sync = false;
  /**
   * @brief If true, a copy kernel will be used to make a contiguous buffer instead of
   *  cudaMemcpyAsync.
   */
  bool use_copy_kernel = false;
  /**
   * @brief Select whether to use the parameter defined in the External Source or
   *  override the mode of operation forcing the copy or no-copy
   */
  ExtSrcNoCopyMode no_copy_mode = ExtSrcNoCopyMode::DEFAULT;
};

/**
 * @brief Provides in-graph access to data fed in from outside of dali.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 * Please note, that it is not allowed to call this concurrently as it
 * may mix the order of inputted data.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend>, virtual public BatchSizeProvider {
  using uptr_tl_type = std::unique_ptr<TensorList<Backend>>;
  using uptr_tv_type = std::unique_ptr<TensorVector<Backend>>;
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;

 public:
  inline explicit ExternalSource(const OpSpec &spec)
      : Operator<Backend>(spec),
        blocking_(spec.GetArgument<bool>("blocking")),
        no_copy_(spec.GetArgument<bool>("no_copy")),
        device_id_(spec.GetArgument<int>("device_id")),
        sync_worker_(device_id_, false) {
    output_name_ = spec.Output(0);
    sync_worker_.WaitForInit();
  }

  inline ~ExternalSource() {
    sync_worker_.ForceStop();
    sync_worker_.Shutdown();
  }

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }

  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template <typename SrcBackend>
  inline void SetDataSource(const TensorList<SrcBackend> &tl, cudaStream_t stream = 0,
                            ExtSrcSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][ExternalSource] SetDataSource", DomainTimeRange::kViolet);
    SetDataSourceHelper(tl, stream, ext_src_setting_mode);
  }

  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template <typename SrcBackend>
  inline void SetDataSource(const vector<Tensor<SrcBackend>> &vect_of_tensors,
                            cudaStream_t stream = 0, ExtSrcSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][ExternalSource] SetDataSource", DomainTimeRange::kViolet);
    TensorVector<SrcBackend> tv(vect_of_tensors.size());
    for (size_t i = 0; i < tv.size(); ++i) {
      tv[i].ShareData(const_cast<Tensor<SrcBackend>*>(&vect_of_tensors[i]));
    }
    SetDataSourceHelper(tv, stream, ext_src_setting_mode);
  }

  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template <typename SrcBackend>
  inline void SetDataSource(const TensorVector<SrcBackend> &tv, cudaStream_t stream = 0,
                            ExtSrcSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][ExternalSource] SetDataSource", DomainTimeRange::kViolet);
    SetDataSourceHelper(tv, stream, ext_src_setting_mode);
  }

  int NextBatchSize() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    return GetStorage().PeekProphet()->ntensor();
  }

  void Advance() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    GetStorage().AdvanceProphet();
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&data = state_] { return !data.empty(); });
    } else {
      if (state_.empty()) {
        DALI_FAIL("No data was provided to the ExternalSource. Make sure to feed it properly.");
      }
    }
    TensorListShape<> shape;
    output_desc.resize(1);
    if (std::is_same<Backend, GPUBackend>::value) {
      output_desc[0].shape = tl_data_.PeekFront()->shape();
      output_desc[0].type = tl_data_.PeekFront()->type();
    } else {
      output_desc[0].shape = tv_data_.PeekFront()->shape();
      output_desc[0].type = tv_data_.PeekFront()->type();
    }
    // unconditionally dissabled, still we can provide share but we don't want to allocate anything
    return false;
  }

  bool CanInferOutputs() const override {
    // shape inference during setup is disabled because it can be calculated during the runtime
    // depending on the input and output
    return false;
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
    // No need to synchronize on copy_to_gpu - it was already synchronized before
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    RecycleBufferHelper(data);
    if (copy_to_gpu) {
      copy_to_storage_events_.Recycle(*copy_to_gpu);
    }
  }

  template<typename SrcBackend, template<typename> class SourceDataType>
  inline std::enable_if_t<!std::is_same<SrcBackend, Backend>::value>
  ShareUserData(const SourceDataType<SrcBackend> &t, cudaStream_t /*stream = 0*/,
                bool /* use_copy_kernel */) {
    DALI_FAIL(make_string("no_copy is supported only for the same data source device type "
                          "as operator. Received: ",
                          std::is_same<SrcBackend, CPUBackend>::value? "CPU" : "GPU",
                          " input for ",
                          std::is_same<Backend, CPUBackend>::value? "CPU" : "GPU",
                          " operator."));
  }

  template <typename SrcBackend, template <typename> class SourceDataType>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const SourceDataType<SrcBackend> &batch, cudaStream_t /*stream = 0*/,
                bool /*use_copy_kernel = false*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({false, true});
    auto tv_elm = tv_data_.GetEmpty();
    // set pinned if needed
    if (batch.is_pinned() != tv_elm.front()->is_pinned()) {
      tv_elm.front()->Reset();
      tv_elm.front()->set_pinned(batch.is_pinned());
    }
    tv_elm.front()->ShareData(const_cast<SourceDataType<CPUBackend> *>(&batch));
    tv_data_.PushBack(tv_elm);
  }

  /**
   * @brief Attempts to share data from tensor vector to tensor list without
   *        an additional copy if the batch is contiguoys.
   *        In case of scattered samples, the data is copied to a contiguous
   *        buffer.
   * @remarks Mixing contiguous and non-contiguous inputs in subsequents calls
   *        is not supported and could lead to data corruption.
   * @param batch source data
   * @param stream CUDA stream use to schedule the copy
   * @param use_copy_kernel If true, a copy kernel will be used to make a
   *        contiguous buffer instead of cudaMemcpyAsync.
   */
  template <typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorVector<SrcBackend> &batch, cudaStream_t stream = 0,
                bool use_copy_kernel = false) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = tl_data_.GetEmpty();
    bool copied_shared_data = false;
    if (batch.IsContiguous()) {
      batch.ShareWith(const_cast<TensorList<Backend> *>(tl_elm.front().get()));
      zero_copy_noncontiguous_gpu_input_ = true;
    } else {
      // it is not contiguous so we need to copy
      tl_elm.front()->Copy(batch, stream, use_copy_kernel);

      std::list<uptr_cuda_event_type> copy_to_storage_event;
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
      CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), stream));
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

  template <typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, cudaStream_t /*stream = 0*/,
                bool /* use_copy_kernel */) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({false, true});
    auto tl_elm = tl_data_.GetEmpty();
    tl_elm.front()->ShareData(const_cast<TensorList<Backend>*>(&batch));
    tl_data_.PushBack(tl_elm);
    zero_copy_noncontiguous_gpu_input_ = true;
  }

  template<typename SrcBackend, template<typename> class SourceDataType, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, CPUBackend>::value>
  CopyUserData(const SourceDataType<SrcBackend> &batch,
               cudaStream_t stream, bool /* sync */, bool /* use_copy_kernel */) {
    std::list<uptr_tv_type> tv_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tv_elm = tv_data_.GetEmpty();
    }
    // set pinned if needed
    if (batch.is_pinned() !=  tv_elm.front()->is_pinned()) {
      tv_elm.front()->Reset();
      tv_elm.front()->set_pinned(batch.is_pinned());
    }
    tv_elm.front()->Copy(batch, stream);
    // if copying from GPU to CPU always synchronize
    if (std::is_same<SrcBackend, GPUBackend>::value) {
      CUDA_CALL(cudaStreamSynchronize(stream));
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tv_data_.PushBack(tv_elm);
      state_.push_back({false, false});
    }
  }

  template<typename SrcBackend, template<typename> class SourceDataType, typename B = Backend>
  inline std::enable_if_t<std::is_same<B, GPUBackend>::value>
  CopyUserData(const SourceDataType<SrcBackend> &batch,
               cudaStream_t stream, bool sync, bool use_copy_kernel) {
    std::list<uptr_cuda_event_type> copy_to_storage_event;
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = tl_data_.GetEmpty();
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
    }
    tl_elm.front()->Copy(batch, stream, use_copy_kernel);
    // record event for:
    // - GPU -> GPU
    // - pinned CPU -> GPU
    if (std::is_same<SrcBackend, GPUBackend>::value || batch.is_pinned()) {
      CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), stream));
    }
    // if copying from non pinned CPU it happens on the stream 0
    if (std::is_same<SrcBackend, CPUBackend>::value && !batch.is_pinned()) {
      CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), 0));
    }
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

  template<typename SrcBackend, template<typename> class SourceDataType>
  inline void SetDataSourceHelper(const SourceDataType<SrcBackend> &batch, cudaStream_t stream = 0,
                                  ExtSrcSettingMode ext_src_setting_mode = {}) {
    bool is_gpu_src = std::is_same<SrcBackend, GPUBackend>::value;
    bool is_gpu_dst = std::is_same<Backend, GPUBackend>::value;
    if (is_gpu_src && !is_gpu_dst) {
      DALI_WARN(
          "Warning: Loading GPU-originated data into CPU ExternalSource operator is discouraged "
          "and might be inefficient.");
    }
    DALI_ENFORCE(
        OperatorBase::max_batch_size_ >= static_cast<int>(batch.ntensor()),
        make_string("Data list provided to ExternalSource needs to have batch_size <= ",
                    OperatorBase::max_batch_size_, ", found ", batch.ntensor(), " samples."));
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.

    bool actual_no_copy = no_copy_;
    switch (ext_src_setting_mode.no_copy_mode) {
      case ExtSrcNoCopyMode::FORCE_COPY:
        actual_no_copy = false;
        break;
      case ExtSrcNoCopyMode::FORCE_NO_COPY:
        actual_no_copy = true;
        break;
      default:
        actual_no_copy = no_copy_;
        break;
    }

    if (actual_no_copy) {
      ShareUserData(batch, stream, ext_src_setting_mode.use_copy_kernel);
    } else {
      CopyUserData(batch, stream, ext_src_setting_mode.sync, ext_src_setting_mode.use_copy_kernel);
    }
    cv_.notify_one();
  }

  string output_name_;
  detail::CachingList<uptr_tl_type> tl_data_;
  detail::CachingList<uptr_tv_type> tv_data_;
  detail::CachingList<uptr_cuda_event_type> copy_to_storage_events_;

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
    /**
     * True if the data that was shared as no_copy required copy regardless.
     * Happens for non-contiguous TensorVector with GPU memory.
     */
    bool copied_shared_data = false;
    /**
     * @brief Actual value of no_copy option used in this call. Always false for CopyUserData(...)
     * and always true for ShareUserData(...)
     */
    bool no_copy = false;
  };

  std::list<ExternalSourceState> state_;

  /*
   * indicates that user provide noncontiguous GPU input with zero copy option so DALI needs
   * to create an internal copy, it is used to raise a warning when the user mixes contiguous and
   * noncontiguous GPU inputs with zero copy what trashed GPU allocated memory
   */
  bool zero_copy_noncontiguous_gpu_input_ = false;

  WorkerThread sync_worker_;

 private:
  using storage_t =
      std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                         detail::CachingList<uptr_tl_type>, detail::CachingList<uptr_tv_type>>;

  template <typename Be = Backend>
  std::enable_if_t<std::is_same<Be, GPUBackend>::value, storage_t &> GetStorage() {
    return tl_data_;
  }

  template <typename Be = Backend>
  std::enable_if_t<!std::is_same<Be, GPUBackend>::value, storage_t &> GetStorage() {
    return tv_data_;
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
