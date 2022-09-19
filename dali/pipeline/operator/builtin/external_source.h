// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
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
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;

  using Operator<Backend>::spec_;

 public:
  inline explicit ExternalSource(const OpSpec &spec)
      : Operator<Backend>(spec),
        blocking_(spec.GetArgument<bool>("blocking")),
        no_copy_(spec.GetArgument<bool>("no_copy")),
        device_id_(spec.GetArgument<int>("device_id")),
        previous_dtype_(DALIDataType::DALI_NO_TYPE),
        ndim_(-1),
        layout_(),
        sync_worker_(device_id_, false, "ExternalSource syncworker") {
    spec.TryGetArgument(dtype_, "dtype");
    if (spec.TryGetArgument(ndim_, "ndim")) {
      DALI_ENFORCE(ndim_ >= 0, make_string("Incorrect number of dimensions (", ndim_,
                   "). Use positive values for tensors or 0 for scalars."));
    }
    spec.TryGetArgument(layout_, "layout");
    InferNdim();
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

  const TensorLayout& layout() const {
    return layout_;
  }

  int ndim() const {
    return ndim_;
  }

  DALIDataType dtype() const {
    return dtype_;
  }

  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template <typename SrcBackend>
  inline void SetDataSource(const vector<Tensor<SrcBackend>> &vect_of_tensors,
                            AccessOrder order = {}, ExtSrcSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][ExternalSource] SetDataSource", DomainTimeRange::kViolet);
    DALI_ENFORCE(vect_of_tensors.size() > 0, "Provided batch cannot be empty.");
    TensorList<SrcBackend> tl(vect_of_tensors.size());
    tl.SetupLike(vect_of_tensors[0]);
    for (int i = 0; i < tl.num_samples(); ++i) {
      tl.SetSample(i, const_cast<Tensor<SrcBackend> &>(vect_of_tensors[i]));
    }
    SetDataSourceHelper(tl, order, ext_src_setting_mode);
  }

  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template <typename SrcBackend>
  inline void SetDataSource(const TensorList<SrcBackend> &tl, AccessOrder order = {},
                            ExtSrcSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][ExternalSource] SetDataSource", DomainTimeRange::kViolet);
    SetDataSourceHelper(tl, order, ext_src_setting_mode);
  }

  int NextBatchSize() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    return tl_data_.PeekProphet()->num_samples();
  }

  void Advance() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    tl_data_.AdvanceProphet();
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  bool HasNdim() {
    return !layout_.empty() || spec_.HasArgument("ndim");
  }

  void InferNdim() {
    if (!layout_.empty()) {
      if (ndim_ != -1) {
        DALI_ENFORCE(ndim_ == layout_.ndim(), make_string("Number of dimensions in the provided "
                     "layout does not match the ndim argument. The arguments provided:",
                     "\n ndim = ", ndim_, ",",
                     "\n layout: \"", layout_, "\"."));
      } else {
        ndim_ = layout_.ndim();
      }
    }
  }

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
    output_desc[0].shape = tl_data_.PeekFront()->shape();
    output_desc[0].type = tl_data_.PeekFront()->type();
    // unconditionally disabled, still we can provide shape but we don't want to allocate anything
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
                          std::is_same<SrcBackend, CPUBackend>::value? "CPU" : "GPU",
                          " input for ",
                          std::is_same<Backend, CPUBackend>::value? "CPU" : "GPU",
                          " operator."));
  }

  template <typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder /* order = {}*/,
                bool /*use_copy_kernel = false*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({false, true});
    auto tl_elm = tl_data_.GetEmpty();
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
  template <typename SrcBackend>
  inline std::enable_if_t<std::is_same<SrcBackend, Backend>::value &&
                          std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder order = {},
                bool use_copy_kernel = false) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = tl_data_.GetEmpty();
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
      tl_elm = tl_data_.GetEmpty();
    }
    // set pinned if needed
    tl_elm.front()->set_order(AccessOrder::host());
    if (batch.is_pinned() !=  tl_elm.front()->is_pinned()) {
      tl_elm.front()->Reset();
      tl_elm.front()->set_pinned(batch.is_pinned());
    }
    tl_elm.front()->Copy(batch, order);
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
      tl_elm = tl_data_.GetEmpty();
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
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

  template <typename SrcBackend>
  inline void ValidateInputData(const TensorList<SrcBackend> &batch) {
    bool is_gpu_src = std::is_same<SrcBackend, GPUBackend>::value;
    bool is_gpu_dst = std::is_same<Backend, GPUBackend>::value;
    if (is_gpu_src && !is_gpu_dst) {
      DALI_WARN(
          "Warning: Loading GPU-originated data into CPU ExternalSource operator is discouraged "
          "and might be inefficient.");
    }

    DALI_ENFORCE(
        OperatorBase::max_batch_size_ >= static_cast<int>(batch.num_samples()),
        make_string("Data list provided to ExternalSource needs to have batch_size <= ",
                    OperatorBase::max_batch_size_, ", found ", batch.num_samples(), " samples."));

    DALI_ENFORCE(
        dtype_ == DALI_NO_TYPE || dtype_ == batch.type(),
        make_string("ExternalSource expected data of type ", TypeTable::GetTypeInfo(dtype_).name(),
        " and got: ", batch.type_info().name()));

    DALI_ENFORCE(previous_dtype_ == DALI_NO_TYPE || previous_dtype_ == batch.type(),
      make_string("Type of the data fed to the external source has changed from the "
                  "previous iteration. Type in the previous iteration was ",
                  TypeTable::GetTypeInfo(previous_dtype_).name(),
                  " and the current type is ", batch.type_info().name(), "."));
    previous_dtype_ = batch.type();

    auto input_ndim = batch.shape().sample_dim();
    if (HasNdim()) {
      DALI_ENFORCE(input_ndim == ndim_,
                   make_string("ExternalSource expected data with ", ndim_, " dimensions and got ",
                     input_ndim, " dimensions."));
    } else if (ndim_ != -1) {
      DALI_ENFORCE(input_ndim == ndim_,
                   make_string("Number of dimensions of the data fed to the external source has "
                      "changed from previous iteration. Dimensionality in the previous "
                      "iteration was ", ndim_, " and the current is ", input_ndim, "."));
    }
    ndim_ = input_ndim;

    if (spec_.HasArgument("layout")) {
      DALI_ENFORCE(layout_ == batch.GetLayout(),
                   make_string("Expected data with layout: \"", layout_,
                     "\" and got: \"", batch.GetLayout(), "\"."));
    } else if (!layout_.empty()) {
      DALI_ENFORCE(layout_ == batch.GetLayout(),
                   make_string("Layout of the data fed to the external source has changed "
                     "from previous iteration. Layout in the previous iteration was \"", layout_,
                     "\" and the current is \"", batch.GetLayout(), "\"."));
    }
    layout_ = batch.GetLayout();
  }

  template<typename SrcBackend>
  inline void SetDataSourceHelper(const TensorList<SrcBackend> &batch, AccessOrder order = {},
                                  ExtSrcSettingMode ext_src_setting_mode = {}) {
    ValidateInputData(batch);

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
      ShareUserData(batch, order, ext_src_setting_mode.use_copy_kernel);
    } else {
      CopyUserData(batch, order, ext_src_setting_mode.sync, ext_src_setting_mode.use_copy_kernel);
    }
    cv_.notify_one();
  }

  string output_name_;
  detail::CachingList<uptr_tl_type> tl_data_;
  detail::CachingList<uptr_cuda_event_type> copy_to_storage_events_;

  std::mutex busy_m_;
  std::condition_variable cv_;
  bool blocking_ = true;
  bool no_copy_ = false;
  int device_id_;
  DALIDataType dtype_ = DALI_NO_TYPE;
  DALIDataType previous_dtype_ = DALI_NO_TYPE;
  int ndim_;
  TensorLayout layout_;

  /*
   * now it only indicates that there is data in the ExternalSource, in the future
   * a per sample metadata could be stored here
   */
  struct ExternalSourceState {
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

  std::list<ExternalSourceState> state_;

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
