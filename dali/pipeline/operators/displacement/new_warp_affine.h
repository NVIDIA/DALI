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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_H_

#include <vector>
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/pipeline/operators/displacement/warp_param_provider.h"

namespace dali {

template <typename Backend>
class NewWarpAffine;

template <int spatial_ndim>
using WarpAffineParams = kernels::warp::mapping_params_t<kernels::AffineMapping<spatial_ndim>>;

template <typename Backend,
          int spatial_ndim,
          typename BorderType>
class WarpAffineParamsProvider
: public WarpParamProvider<Backend, spatial_ndim, WarpAffineParams<spatial_ndim>, BorderType> {
 protected:
  using MappingParams = WarpAffineParams<spatial_ndim>;
  using Base = WarpParamProvider<Backend, spatial_ndim, MappingParams, BorderType>;
  using Workspace = typename Base::Workspace;
  using Base::ws_;
  using Base::spec_;
  using Base::params_gpu_;
  using Base::params_cpu_;
  using Base::num_samples_;

  void SetParams() override {
    if (spec_->NumRegularInput() >= 2) {
      if (ws_->template InputIsType<GPUBackend>(1)) {
        UseInput(ws_->template Input<GPUBackend>(1));
      } else {
        UseInput(ws_->template Input<CPUBackend>(1));
      }
    } else {
      std::vector<float> matrix = spec_->template GetArgument<std::vector<float>>("matrix");

      DALI_ENFORCE(matrix.size() == spatial_ndim*(spatial_ndim+1));

      MappingParams M;
      int k = 0;
      for (int i = 0; i < spatial_ndim; i++)
        for (int j = 0; j < spatial_ndim+1; j++, k++)
          M.transform(i, j) = matrix[k];

      auto *params = this->AllocParams(kernels::AllocType::Host);
      for (int i = 0; i < params_cpu_.shape[0]; i++)
        params[i] = M;
    }
  }

  template <typename InputType>
  void CheckInput(const InputType &input) {
    DALI_ENFORCE(input.type().id() == DALI_FLOAT);
    DALI_ENFORCE(input.shape().num_samples() == num_samples_,
      "Internal error: mismatched number of samples");

    kernels::TensorShape<2> shape = { spatial_ndim, spatial_ndim+1 };
    for (int i = 0; i < num_samples_; i++) {
      DALI_ENFORCE(input.shape()[i] == shape);
    }
  }

  void UseInput(const TensorList<CPUBackend> &input) {
    CheckInput(input);

    params_cpu_.data = static_cast<const MappingParams *>(input.raw_data());
    params_cpu_.shape = { num_samples_ };
  }

  void UseInput(const TensorVector<CPUBackend> &input) {
    CheckInput(input);

    if (!input.IsContiguous()) {
      auto *params = this->AllocParams(kernels::AllocType::Host);
      for (int i = 0; i < num_samples_; i++) {
        auto &tensor = input[i];
        params[i] = *static_cast<const MappingParams *>(input[i].raw_data());
      }
    } else {
      params_cpu_.data = static_cast<const MappingParams *>(input[0].raw_data());
      params_cpu_.shape = { num_samples_ };
    }
  }

  void UseInput(const TensorList<GPUBackend> &input) {
    CheckInput(input);

    params_gpu_.data = static_cast<const MappingParams *>(input.raw_data());
    params_gpu_.shape = { num_samples_ };
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_H_
