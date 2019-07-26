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

#ifndef DALI_KERNELS_IMGPROC_WARP_VARIABLE_SIZE_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_WARP_VARIABLE_SIZE_IMPL_CUH_

#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/block_warp.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"

namespace dali {
namespace kernels {
namespace warp {


template <DALIInterpType interp_type, typename Mapping,
         int ndim, typename OutputType, typename InputType,
         typename BorderValue>
__global__ void BatchWarpVariableSize(
    const SampleDesc<ndim, OutputType, InputType> *samples,
    const BlockDesc<ndim> *blocks,
    const mapping_params_t<Mapping> *mapping,
    BorderValue border) {

  auto block = blocks[blockIdx.x];
  auto sample = samples[block.sample_idx];
  BlockWarp<interp_type, Mapping, ndim, OutputType, InputType, BorderValue>(
    sample, block, Mapping(mapping[block.sample_idx]), border);
}

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_VARIABLE_SIZE_IMPL_CUH_
