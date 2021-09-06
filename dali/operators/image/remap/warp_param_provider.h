// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_WARP_PARAM_PROVIDER_H_
#define DALI_OPERATORS_IMAGE_REMAP_WARP_PARAM_PROVIDER_H_

#include <cassert>
#include <vector>
#include <string>

#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/core/mm/memory.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/common/copy.h"

namespace dali {

class InterpTypeProvider {
 public:
  span<const DALIInterpType> InterpTypes() const {
    return make_span(interp_types_);
  }

 protected:
  void SetInterp(const OpSpec &spec, const ArgumentWorkspace &ws, int num_samples) {
    interp_types_.clear();
    if (spec.HasTensorArgument("interp_type")) {
      auto &tensor_vector = ws.ArgumentInput("interp_type");
      int n = tensor_vector.shape().num_samples();
      DALI_ENFORCE(n == 1 || n == num_samples,
        "interp_type must be a single value or contain one value per sample");
      interp_types_.resize(n);
      for (int i = 0; i < n; i++)
        interp_types_[i] = tensor_vector[i].data<DALIInterpType>()[0];
    } else {
      interp_types_.resize(1, spec.template GetArgument<DALIInterpType>("interp_type"));
    }

    for (size_t i = 0; i < interp_types_.size(); i++) {
      DALI_ENFORCE(interp_types_[i] == DALI_INTERP_NN || interp_types_[i] == DALI_INTERP_LINEAR,
        "Only nearest and linear interpolation is supported");
    }
  }

  InterpTypeProvider() = default;
  std::vector<DALIInterpType> interp_types_;
};

template <typename BorderType>
class BorderTypeProvider {
 public:
  BorderType Border() const {
    return border_;
  }
 protected:
  void SetBorder(const OpSpec &spec) {
    float fborder;
    int iborder;
    if (spec.TryGetArgument(fborder, "fill_value"))
      border_ = ConvertSat<BorderType>(fborder);
    else if (spec.TryGetArgument(iborder, "fill_value"))
      border_ = ConvertSat<BorderType>(iborder);
  }
  BorderType border_ = {};
};

template <>
inline void BorderTypeProvider<kernels::BorderClamp>::SetBorder(const OpSpec &spec) {
}

/** @brief Provides warp parameters
 *
 *  The classes derived from WarpParamProvider interpret the OpSpec and
 *  Workspace arguments and inputs and provide warp parameters, output sizes,
 *  border value, etc
 *
 *  Usage:
 *  - In operator setup: SetContext, Setup
 *  - In operator run: SetContext, GetParams[GPU/CPU]
 *  Overriding:
 *  - Provide SetParams and InferShape (if supported)
 */
template <typename Backend, int spatial_ndim, typename MappingParams, typename BorderType>
class WarpParamProvider : public InterpTypeProvider, public BorderTypeProvider<BorderType> {
 public:
  using SpatialShape = TensorShape<spatial_ndim>;
  using Workspace = workspace_t<Backend>;

  virtual ~WarpParamProvider() = default;

  void SetContext(const OpSpec &spec, const Workspace &ws) {
    spec_ = &spec;
    ws_ = &ws;
    num_samples_ = NumSamples(ws);
  }

  /** @brief Prepares parameters and output sizes for a warp operator
   *
   * This function sets sizes and shapes in four steps:
   * 1. Use explicitly provided sizes or copy from input
   * 2. Set transform parameters - may depend on sizes specified in 1
   * 3. Infer sizes based on params calculated in step 2, if not already set in 1
   * 4. Adjust parameters, if required, after shape inference.
   * Steps 1 and 3 are mutually exclusive.
   *
   * If different scheme is required, the derived class must override this method.
   *
   * Examples:
   * Size-dependent transform: rotate and fit to canvas
   * Transform-dependent size: canvas resized to fit rotated image
   */
  virtual void Setup() {
    assert(ws_ && spec_ && "Use SetContext before calling Setup");
    ResetParams();
    // Step 1: Check if the sizes are specified explicitly or copied
    // from the input size. These sizes do not depend on the
    // transform params, so they should be used first.
    bool infer_size = !SetOutputSizes();
    // Step 2: Set the parameters. which _may_ depend on explicitly set sizes
    SetParams();
    // Step 3: If the operator must infer the output size based
    // on the params, then this size inference must obviously
    // follow SetParams.
    if (infer_size)
      InferSize();
    // Step 4: Adjust parameters after shape inference
    AdjustParams();

    // Interpolation type and border can be set at any time
    this->SetInterp(*spec_, *ws_, num_samples_);
    this->SetBorder(*spec_);
  }

  virtual bool KeepOriginalSize() const {
    return !HasExplicitSize() && !ShouldInferSize();
  }

  virtual bool ShouldInferSize() const {
    bool resize_to_fit = false;
    return spec_->TryGetArgument(resize_to_fit, "infer_size") && resize_to_fit;
  }

  virtual bool HasExplicitSize() const {
    return spec_->ArgumentDefined(size_arg_name_);
  }

  virtual bool HasExplicitPerSampleSize() const {
    return spec_->HasTensorArgument(size_arg_name_);
  }

  span<const SpatialShape> OutputSizes() const {
    return make_span(out_sizes_);
  }

  /** @brief Gets the mapping parameters in GPU memory
   *
   *  If GPU tensor is empty, but CPU is not, an asyncrhonous copy is scheduled
   *  on the stream associated with current workspace.
   */
  TensorView<StorageGPU, const MappingParams, 1> ParamsGPU() {
    if (!params_gpu_.data && params_cpu_.data) {
      auto *p = AllocParams<mm::memory_kind::device>(params_cpu_.num_elements());
      auto tmp = make_tensor_gpu(p, params_cpu_.shape);
      kernels::copy(tmp, params_cpu_, GetStream());
    }
    return params_gpu_;
  }

  /** @brief Gets the mapping parameters in GPU memory
   *
   *  If CPU tensor is empty, but GPU is not, a copy is scheduled
   *  on the stream associated with current workspace and the calling thread
   *  is synchronized with the stream.
   */
  TensorView<StorageCPU, const MappingParams, 1> ParamsCPU() {
    if (!params_cpu_.data && params_gpu_.data) {
      auto *p = AllocParams<mm::memory_kind::host>(params_gpu_.num_elements());
      auto tmp = make_tensor_cpu(p, params_cpu_.shape);
      cudaStream_t stream = GetStream();
      kernels::copy(tmp, params_gpu_, stream);
      CUDA_CALL(cudaStreamSynchronize(stream));
    }
    return params_cpu_;
  }

 protected:
  inline cudaStream_t GetStream() const {
    return ws_ && ws_->has_stream() ? ws_->stream() : 0;
  }

  static inline int NumSamples(const Workspace &ws) {
    return ws.template InputRef<Backend>(0).shape().num_samples();
  }

  virtual void ResetParams() {
    params_gpu_ = {};
    params_cpu_ = {};
  }

  virtual void SetParams() {
  }

  virtual void AdjustParams() {
  }

  virtual void ValidateOutputSizes() {}

  virtual void GetUniformOutputSize(SpatialShape &out_size) const {
    assert(HasExplicitSize() && !HasExplicitPerSampleSize());
    std::vector<float> out_size_f = spec_->template GetArgument<std::vector<float>>("size");
    DALI_ENFORCE(static_cast<int>(out_size_f.size()) == spatial_ndim,
      "output_size must specify same number of dimensions as the input (excluding channels)");
    for (int d = 0; d < spatial_ndim; d++) {
      float s = out_size_f[d];
      DALI_ENFORCE(s > 0, "Output size must be positive");
      out_size[d] = std::max<int>(std::roundf(s), 1);
    }
  }

  template <typename T>
  void GetTypedPerSampleSize(std::vector<SpatialShape> &out_sizes,
                             const TensorListView<StorageCPU, T> &shape_list) const {
    const auto &shape = shape_list.shape;
    const int N = num_samples_;

    DALI_ENFORCE(is_uniform(shape), "Output sizes must be passed as uniform Tensor List.");
    DALI_ENFORCE(
        (shape.num_samples() == N && shape[0] == TensorShape<>(spatial_ndim)) ||
            (shape.num_samples() == 1 && (shape[0] == TensorShape<>(N, spatial_ndim) ||
                                          shape[0] == TensorShape<>(N * spatial_ndim))),
        "Output sizes must either be a batch of `dim`-sized tensors, flat array of size "
        "num_samples*dim or one 2D tensor of shape {num_samples, dim}.");

    out_sizes.resize(N);
    if (shape.num_samples() == N) {
      for (int i = 0; i < N; i++)
        for (int d = 0; d < spatial_ndim; d++)
          out_sizes[i][d] = shape_list.data[i][d];
    } else {
      for (int i = 0; i < N; i++)
        for (int d = 0; d < spatial_ndim; d++)
          out_sizes[i][d] = shape_list.data[0][i*N + d];
    }
  }

  virtual void GetExplicitPerSampleSize(std::vector<SpatialShape> &out_sizes) const {
    assert(HasExplicitPerSampleSize());
    const auto &tensor_vector = ws_->ArgumentInput(size_arg_name_);
    TYPE_SWITCH(tensor_vector.type().id(), type2id, shape_t,
      (int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, float),
      (GetTypedPerSampleSize(out_sizes, view<const shape_t>(tensor_vector))),
      (DALI_FAIL(make_string("Warp: Unsupported argument type for \"", size_arg_name_, "\": ",
        tensor_vector.type().id())))
    );  // NOLINT
  }

  void SetExplicitSize() {
    if (HasExplicitPerSampleSize()) {
      GetExplicitPerSampleSize(out_sizes_);
    } else {
      assert(HasExplicitSize());
      SpatialShape out_shape;
      GetUniformOutputSize(out_shape);
      out_sizes_.resize(num_samples_);
      for (auto &s : out_sizes_)
        s = out_shape;
    }
  }

  virtual bool SetOutputSizes() {
    decltype(auto) input_shape = ws_->template InputRef<Backend>(0).shape();
    const int N = input_shape.num_samples();
    out_sizes_.resize(N);
    SpatialShape scalar_size;

    if (HasExplicitSize()) {
      SetExplicitSize();
      return true;
    } else if (ShouldInferSize()) {
      return false;
    } else {
      assert(KeepOriginalSize());
      for (int i = 0; i < N; i++) {
        out_sizes_[i] = input_shape[i].template first<spatial_ndim>();
      }
      return true;
    }
  }

  virtual void InferSize() {
    DALI_FAIL("This operator does not support size inference.");
  }

  /** @brief Allocates num_samples_ MappingParams objects in memory specified by alloc  */
  template <typename MemoryKind>
  MappingParams *AllocParams() {
    return AllocParams<MemoryKind>(num_samples_);
  }

  template <typename MemoryKind>
  auto &SelectParamView() {
    return SelectParamView(static_cast<MemoryKind*>(nullptr));
  }

  inline auto &SelectParamView(mm::memory_kind::device *) {
    return params_gpu_;
  }

  inline auto &SelectParamView(...) {
    return params_cpu_;
  }

  /** @brief Allocates count MappingParams objects in memory specified by alloc  */
  template <typename MemoryKind>
  MappingParams *AllocParams(int count) {
    param_mem_.Reserve<MemoryKind>(count * sizeof(MappingParams));
    auto scratch = param_mem_.GetScratchpad();
    auto tmp = scratch.template AllocTensor<MemoryKind, MappingParams, 1>(count);
    SelectParamView<MemoryKind>() = tmp;
    return tmp.data;
  }

  // can be overwritten by a derived class
  std::string size_arg_name_ = "size";
  const OpSpec *spec_ = nullptr;
  const Workspace *ws_ = nullptr;
  int num_samples_ = 0;

  std::vector<SpatialShape> out_sizes_;
  TensorView<StorageGPU, const MappingParams, 1> params_gpu_;
  TensorView<StorageCPU, const MappingParams, 1> params_cpu_;
  kernels::ScratchpadAllocator param_mem_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_WARP_PARAM_PROVIDER_H_
