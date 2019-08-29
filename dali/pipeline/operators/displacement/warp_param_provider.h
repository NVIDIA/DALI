// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_PARAM_PROVIDER_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_PARAM_PROVIDER_H_

#include <cassert>
#include <vector>
#include <string>

#include "dali/kernels/tensor_view.h"
#include "dali/kernels/alloc.h"
#include "dali/pipeline/operators/operator.h"
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
      auto &tensor = ws.ArgumentInput("interp_type");
      int n = tensor.shape()[0];
      DALI_ENFORCE(n == 1 || n == num_samples,
        "interp_type must be a single value or contain one value per sample");
      auto *data = tensor.template data<DALIInterpType>();
      interp_types_.resize(n);

      for (int i = 0; i < n; i++)
        interp_types_[i] = data[i];
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
    if (spec.TryGetArgument(fborder, "border"))
      border_ = ConvertSat<BorderType>(fborder);
    else if (spec.TryGetArgument(iborder, "border"))
      border_ = ConvertSat<BorderType>(iborder);
  }
  BorderType border_ = {};
};

template <>
inline void BorderTypeProvider<kernels::BorderClamp>::SetBorder(const OpSpec &spec) {
}

/// @brief Provides warp parameters
///
/// The classes derived from WarpParamProvider interpret the OpSpec and
/// Workspace arguments and inputs and provide warp parameters, output sizes,
/// border value, etc
///
/// Usage:
/// In operator setup: SetContext, Setup
/// In operator run: SetContext, GetParams[GPU/CPU]
/// Overriding:
/// Provide SetParams and InferShape (if supported)
template <typename Backend, int spatial_ndim, typename MappingParams, typename BorderType>
class WarpParamProvider : public InterpTypeProvider, public BorderTypeProvider<BorderType> {
 public:
  using SpatialShape = kernels::TensorShape<spatial_ndim>;
  using Workspace = workspace_t<Backend>;

  virtual ~WarpParamProvider() = default;

  void SetContext(const OpSpec &spec, const Workspace &ws) {
    spec_ = &spec;
    ws_ = &ws;
    num_samples_ = NumSamples(ws);
  }

  virtual void Setup() {
    assert(ws_ && spec_ && "Use SetContext before calling Setup");
    ResetParams();
    bool infer_size = !SetOutputSizes();
    SetParams();
    if (infer_size)
      InferSize();
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
    return spec_->HasArgument(size_arg_name_);
  }

  virtual bool HasExplicitPerSampleSize() const {
    return spec_->HasTensorArgument(size_arg_name_);
  }

  span<const SpatialShape> OutputSizes() const {
    return make_span(out_sizes_);
  }

  /// @brief Gets the mapping parameters in GPU memory
  ///
  /// If GPU tensor is empty, but CPU is not, an asyncrhonous copy is scheduled
  /// on the stream associated with current workspace.
  kernels::TensorView<kernels::StorageGPU, const MappingParams, 1> ParamsGPU() {
    if (!params_gpu_.data && params_cpu_.data) {
      auto *p = AllocParams(kernels::AllocType::GPU, params_cpu_.num_elements());
      auto tmp = make_tensor_gpu(p, params_cpu_.shape);
      kernels::copy(tmp, params_cpu_, GetStream());
    }
    return params_gpu_;
  }

  /// @brief Gets the mapping parameters in GPU memory
  ///
  /// If CPU tensor is empty, but GPU is not, a copy is scheduled
  /// on the stream associated with current workspace and the calling thread
  /// is synchronized with the stream.
  kernels::TensorView<kernels::StorageCPU, const MappingParams, 1> ParamsCPU() {
    if (!params_cpu_.data && params_gpu_.data) {
      auto *p = AllocParams(kernels::AllocType::Host, params_gpu_.num_elements());
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

  virtual void GetExplicitPerSampleSize(std::vector<SpatialShape> &out_sizes) const {
    assert(HasExplicitPerSampleSize());
    const Tensor<CPUBackend> &tensor = ws_->ArgumentInput(size_arg_name_);
    auto tv = view<const int>(tensor);
    const int N = num_samples_;
    DALI_ENFORCE(
      tv.shape == kernels::TensorShape<>(N, spatial_ndim) ||
      tv.shape == kernels::TensorShape<>(N * spatial_ndim),
      "Output sizes must either be a flat array of size num_samples*dim "
      "or a 2D tensor of size num_samples x dim");

    out_sizes.resize(N);
    for (int i = 0; i < N; i++)
      for (int d = 0; d < spatial_ndim; d++)
        out_sizes[i][d] = tv.data[i*N + d];
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

  MappingParams *AllocParams(kernels::AllocType alloc) {
    return AllocParams(alloc, num_samples_);
  }

  MappingParams *AllocParams(kernels::AllocType alloc, int count) {
    param_mem_.Reserve(alloc, count * sizeof(MappingParams));
    auto scratch = param_mem_.GetScratchpad();
    if (alloc == kernels::AllocType::GPU) {
      auto tmp = scratch.template AllocTensor<kernels::AllocType::GPU, MappingParams, 1>(count);
      params_gpu_ = tmp;
      return tmp.data;
    } else if (alloc == kernels::AllocType::Host) {
      auto tmp = scratch.template AllocTensor<kernels::AllocType::Host, MappingParams, 1>(count);
      params_cpu_ = tmp;
      return tmp.data;
    } else if (alloc == kernels::AllocType::Pinned) {
      auto tmp = scratch.template AllocTensor<kernels::AllocType::Pinned, MappingParams, 1>(count);
      params_cpu_ = tmp;
      return tmp.data;
    } else {
      assert(!"Unsupported allocation type requested");
      return nullptr;
    }
  }

  // can be overwritten by a derived class
  std::string size_arg_name_ = "size";
  const OpSpec *spec_ = nullptr;
  const Workspace *ws_ = nullptr;
  int num_samples_ = 0;

  std::vector<SpatialShape> out_sizes_;
  BorderType border_;
  kernels::TensorView<kernels::StorageGPU, const MappingParams, 1> params_gpu_;
  kernels::TensorView<kernels::StorageCPU, const MappingParams, 1> params_cpu_;
  kernels::ScratchpadAllocator param_mem_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_PARAM_PROVIDER_H_
