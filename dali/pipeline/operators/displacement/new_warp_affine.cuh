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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_

#include "dali/pipeline/operators/displacement/warp_impl.cuh"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/imgproc/warp/affine.h"

namespace dali {

template <typename Backend>
class NewWarpAffine;

template <>
class NewWarpAffine<GPUBackend> : public Warp<GPUBackend, NewWarpAffine<GPUBackend>> {
 public:
  using Base = Warp<GPUBackend, NewWarpAffine<GPUBackend>>;
  using Base::Base;

  template <int ndim, typename OutputType, typename InputType>
  using KernelType = kernels::WarpGPU<
    kernels::AffineMapping<ndim>, ndim, OutputType, InputType, InputType>;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_NEW_WARP_AFFINE_CUH_
