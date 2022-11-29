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

#include "dali/core/access_order.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/input/caching_list.h"
#include "dali/pipeline/input/input_operator.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"
#include "dali/core/common.h"

namespace dali {


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
class ExternalSource : public InputOperator<Backend>, virtual public BatchSizeProvider {
  using InputOperator<Backend>::spec_;
  using InputOperator<Backend>::state_;
  using InputOperator<Backend>::GetOutputDataQueue;
  using InputOperator<Backend>::busy_m_;
  using InputOperator<Backend>::HasData;
  using InputOperator<Backend>::uptr_tl_type;
  using InputOperator<Backend>::uptr_cuda_event_type;

 public:
  inline explicit ExternalSource(const OpSpec &spec)
      : InputOperator<Backend>(spec),
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

  virtual ~ExternalSource() {
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
    return GetOutputDataQueue().PeekProphet()->num_samples();
  }

  void Advance() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    GetOutputDataQueue().AdvanceProphet();
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

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&]() { return HasData(); });
    } else {
      if (!HasData()) {
        DALI_FAIL("No data was provided to the ExternalSource. Make sure to feed it properly.");
      }
    }
    TensorListShape<> shape;
    output_desc.resize(1);
    output_desc[0].shape = GetOutputDataQueue().PeekFront()->shape();
    output_desc[0].type = GetOutputDataQueue().PeekFront()->type();
    // unconditionally disabled, still we can provide shape, but we don't want to allocate anything
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

  void RunImpl(Workspace &ws) override;


  template <typename SrcBackend>
  inline void ValidateInputData(const TensorList<SrcBackend> &batch) {
    const bool is_gpu_src = std::is_same<SrcBackend, GPUBackend>::value;
    const bool is_gpu_dst = std::is_same<Backend, GPUBackend>::value;
    if (is_gpu_src && !is_gpu_dst) {
      DALI_WARN(
          "Warning: Loading GPU-originated data into CPU ExternalSource operator is discouraged "
          "and might be inefficient.");
    }

    DALI_ENFORCE(
        OperatorBase::max_batch_size_ >= static_cast<int>(batch.num_samples()),
        make_string("Data list provided to ExternalSource needs to have batch_size <= ",
                    OperatorBase::max_batch_size_, ", found ", batch.num_samples(), " samples."));

    DALI_ENFORCE(batch.num_samples() > 0,
                 "ExternalSource expects non-empty batches to be provided as the input. Got batch "
                 "with 0 samples.");

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
      this->ShareUserData(batch, order, ext_src_setting_mode.use_copy_kernel);
    } else {
      this->CopyUserData(batch, order, ext_src_setting_mode.sync, ext_src_setting_mode.use_copy_kernel);
    }
    cv_.notify_one();
  }


  string output_name_;

  std::condition_variable cv_;
  bool blocking_ = true;
  bool no_copy_ = false;
  int device_id_;
  DALIDataType dtype_ = DALI_NO_TYPE;
  DALIDataType previous_dtype_ = DALI_NO_TYPE;
  int ndim_;
  TensorLayout layout_;

  WorkerThread sync_worker_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
