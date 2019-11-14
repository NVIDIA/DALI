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

#ifndef DALI_KERNELS_IMGPROC_WARP_UNIFORM_SIZE_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_WARP_UNIFORM_SIZE_IMPL_CUH_

#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/block_warp.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace warp {

template <typename Mapping,
         int ndim, typename OutputType, typename InputType,
         typename BorderType>
__global__ void BatchWarpUniformSize(
    const SampleDesc<ndim, OutputType, InputType> *samples,
    ivec<ndim> output_size,
    ivec<ndim> block_size,
    int z_blocks_per_sample_shift,
    const mapping_params_t<Mapping> *mapping,
    BorderType border) {
  BlockDesc<ndim> block;
  ivec<ndim> block_idx(0);
  block_idx.x = blockIdx.x;
  block_idx.y = blockIdx.y;
  if (z_blocks_per_sample_shift) {
    block.sample_idx = blockIdx.z >> z_blocks_per_sample_shift;
    int block_in_sample = blockIdx.z & ((1 << z_blocks_per_sample_shift) - 1);
    block_idx[2] = block_in_sample;
  } else {
    block.sample_idx = blockIdx.z;
  }
  block.start = block_size * block_idx;
  block.end = min(block.start + block_size, output_size);
  auto sample = samples[block.sample_idx];
  VALUE_SWITCH(sample.interp, interp_const, (DALI_INTERP_NN, DALI_INTERP_LINEAR), (
    BlockWarp<interp_const, Mapping, OutputType, InputType, BorderType>(
      sample, block, Mapping(mapping[block.sample_idx]), border)),
    (assert(!"Interpolation type not supported")));
}

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_UNIFORM_SIZE_IMPL_CUH_
