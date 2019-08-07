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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_

#include "dali/pipeline/operators/displacement/new_warp_affine.h"
#include "dali/pipeline/operators/displacement/warp_impl.cuh"
#include "dali/kernels/imgproc/warp/affine.h"

namespace dali {

template <size_t r, size_t c>
__global__ void BatchTranspose(mat<c, r> *out, const mat<r, c> *in, int n) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  auto m = in[i];
  out[i] = m.T();
}

template <typename Backend>
class NewWarpAffine;

template <>
class NewWarpAffine<GPUBackend> : public Warp<GPUBackend, NewWarpAffine<GPUBackend>> {
 public:
  using Base = Warp<GPUBackend, NewWarpAffine<GPUBackend>>;
  using Base::Base;

  template <int ndim>
  using Mapping = kernels::AffineMapping<ndim>;

  template <int ndim, typename OutputType, typename InputType, typename BorderType>
  using KernelType = kernels::WarpGPU<
    Mapping<ndim>, ndim, OutputType, InputType, BorderType>;

  template <int spatial_ndim, typename BorderType>
  using ParamProvider = WarpAffineParamsProvider<GPUBackend, spatial_ndim, BorderType>;

  template <int spatial_ndim, typename BorderType>
  auto CreateParamProvider() {
    return std::make_unique<ParamProvider<spatial_ndim, BorderType>>();
  }

};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_
