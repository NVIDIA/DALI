// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/tensor_view.h"
#include "dali/kernels/alloc.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend, int spatial_ndim, typename MappingParams>
class WarpParamProvider {
 public:
  using SpatialShape = kernels::TensorShape<spatial_ndim>;
  using Workspace = dali::Workspace<Backend>;

  using Storage = std::conditional<
      std::is_same<Backend, GPUBackend>::value,
      kernels::StorageGPU, kernels::StorageCPU>;

  virtual ~WarpParamProvider() = default;

  OpSpec &Spec() { return *spec_; }

  virtual void Setup(OpSpec &spec, Workspace &ws) {
    spec_ = &spec;
    SetOutputSizes(ws);
  }

  virtual bool GetUniformOutputSize(SpatialShape &out_size) {
    std::vector<float> out_size_f;
    if (!spec_->TryGetArgument(out_size_f, "output_size"))
      return false;

    DALI_ENFORCE(static_cast<int>(out_size_f.size()) == spatial_ndim,
      "output_size must specify same number of dimensions as the input (excluding channels)");
    for (int d = 0; d < spatial_ndim; d++) {
      float s = out_size_f[d];
      DALI_ENFORCE(s > 0, "Output size must be positive");
      out_size[d] = std::max<int>(std::roundf(s), 1);
    }
    return true;
  }

  virtual bool KeepOriginalSize() const {
    return
      !spec_->HasArgument("output_size") &&
      ResizeToFit();
  }

  virtual bool ResizeToFit() const {
    bool resize_to_fit = false;
    return spec_->TryGetArgument(resize_to_fit, "resize_to_fit") && resize_to_fit;
  }

  span<kernels::TensorShape<spatial_ndim>> GetOutputSizes() {
    return make_span(output_sizes_);
  }

  kernels::TensorView<kernels::StorageGPU, MappingParams, 1> GetParams() {
    return params_gpu_;
  }

  void SetInterp(Workspace &ws) {
    interp_types_.clear();
    if (Spec().HasTensorArgument("interp_type")) {
      int num_samples = ws.Input<Backend>(0).shape().num_samples();
      auto &tensor = ws.ArgumentInput("interp_type");
      int n = tensor.shape()[0];
      DALI_ENFORCE(n == 1 || n == num_samples,
        "interp_type must be a single value or contain one value per sample");
      auto *data = tensor.template data<DALIInterpType>();
      interp_types_.resize(n);

      for (int i = 0; i < n; i++)
        interp_types_[i] = data[i];
    } else {
      interp_types_.resize(1, Spec().template GetArgument<DALIInterpType>("interp_type"));
    }

    for (size_t i = 0; i < interp_types_.size(); i++) {
      DALI_ENFORCE(interp_types_[i] == DALI_INTERP_NN || interp_types_[i] == DALI_INTERP_LINEAR,
        "Only nearest and linear interpolation is supported");
    }
  }


 protected:
  virtual void SetOutputSizes(Workspace &ws) {
    const int N = ws.Inputs<Backend>(0).shape().num_samples();
    output_sizes_.resize(N);
    GetScalarOutputSize();

    if (size == kernels::TensorShape<spatial_ndim>()) {
      for (int i = 0; i < N; i++) {
        output_sizes_[i] = input_.shape[i].template first<spatial_ndim>();
      }
    }
  }


  virtual void CalculateSizetoFit() {
    FAIL("This method is not implemented");
  }

  OpSpec *spec_ = nullptr;
  std::vector<kernels::TensorShape<spatial_ndim>> output_sizes_;
  std::vector<DALIInterpType> interp_types_;
  kernels::TensorView<Storage, MappingParams, 1> params_;

};
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_PARAM_PROVIDER_H_
