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

#ifndef DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_
#define DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_

#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"

namespace dali {
namespace kernels {
namespace warp {

template <typename Mapping, std::size_t dim>
DALI_HOST_DEV
enable_if_t<is_fp_mapping<Mapping>::value, vec<dim>> map_coords(const Mapping &m, ivec<dim> pos) {
  return m(pos + 0.5f);
}

template <typename Mapping, std::size_t dim>
DALI_HOST_DEV
enable_if_t<!is_fp_mapping<Mapping>::value, ivec<dim>> map_coords(const Mapping &m, ivec<dim> pos) {
  return m(pos);
}

template <DALIInterpType interp_type, typename Mapping,
          int ndim, typename OutputType, typename InputType,
          typename BorderValue>
__device__ void BlockWarp(
    SampleDesc<2, OutputType, InputType> sample, BlockDesc<2> block,
    Mapping mapping, BorderValue border) {
  // Get the data pointers - un-erase type
  OutputType *__restrict__ output_data = sample.output;
  const InputType *__restrict__ input_data = sample.input;
  // Create input and output surfaces
  const Surface2D<OutputType> out = {
      output_data, sample.out_size.x, sample.out_size.y, sample.channels,
      sample.out_strides.x, sample.out_strides.y, 1
  };
  const Surface2D<const InputType> in = {
      input_data, sample.in_size.x, sample.in_size.y, sample.channels,
      sample.in_strides.x, sample.in_strides.y, 1
  };
  // ...and a sampler
  const auto sampler = make_sampler<interp_type>(in);

  // Run this HW block of threads over the logical block
  for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
    for (int x = block.start.x + threadIdx.x; x < block.end.x; x += blockDim.y) {
      auto src = map_coords(mapping, ivec2(x, y));
      sampler(&out(x, y), src, border);
    }
  }
}

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_
