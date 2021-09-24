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

#ifndef DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_
#define DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_

#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/kernels/imgproc/warp/map_coords.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace warp {


template <int static_channels,
          DALIInterpType interp_type, typename Mapping,
          typename OutputType, typename InputType,
          typename BorderType>
__device__ void BlockWarpChannels(
    SampleDesc<2, OutputType, InputType> sample, BlockDesc<2> block,
    Mapping mapping, BorderType border) {
  const int channels = static_channels < 0 ? sample.channels : static_channels;
  // Get the data pointers - un-erase type
  OutputType *__restrict__ output_data = sample.output;
  const InputType *__restrict__ input_data = sample.input;
  // Create input and output surfaces
  const Surface2D<OutputType> out = {
      output_data, sample.out_size, channels,
      sample.out_strides, 1
  };
  const Surface2D<const InputType> in = {
      input_data, sample.in_size, channels,
      sample.in_strides, 1
  };
  // ...and a sampler
  const auto sampler = make_sampler<interp_type>(in);

  // Run this HW block of threads over the logical block
  for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
    for (int x = block.start.x + threadIdx.x; x < block.end.x; x += blockDim.x) {
      auto src = map_coords(mapping, ivec2(x, y));
      sampler(&out(x, y), src, border);
    }
  }
}

template <int static_channels,
          DALIInterpType interp_type, typename Mapping,
          typename OutputType, typename InputType,
          typename BorderType>
__device__ void BlockWarpChannels(
    SampleDesc<3, OutputType, InputType> sample, BlockDesc<3> block,
    Mapping mapping, BorderType border) {
  const int channels = static_channels < 0 ? sample.channels : static_channels;
  // Get the data pointers - un-erase type
  OutputType *__restrict__ output_data = sample.output;
  const InputType *__restrict__ input_data = sample.input;
  // Create input and output surfaces
  const Surface3D<OutputType> out = {
      output_data, sample.out_size, channels,
      sample.out_strides, 1
  };
  const Surface3D<const InputType> in = {
      input_data, sample.in_size, channels,
      sample.in_strides, 1
  };
  // ...and a sampler
  const auto sampler = make_sampler<interp_type>(in);

  // Run this HW block of threads over the logical block
  for (int z = block.start.z + threadIdx.z; z < block.end.z; z += blockDim.z) {
    for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
      for (int x = block.start.x + threadIdx.x; x < block.end.x; x += blockDim.x) {
        auto src = map_coords(mapping, ivec3(x, y, z));
        sampler(&out(x, y, z), src, border);
      }
    }
  }
}

template <DALIInterpType interp_type, typename Mapping,
          typename OutputType, typename InputType,
          typename BorderType, int spatial_ndim>
__device__ void BlockWarp(
    SampleDesc<spatial_ndim, OutputType, InputType> sample, BlockDesc<spatial_ndim> block,
    Mapping mapping, BorderType border) {
  VALUE_SWITCH(sample.channels, static_channels, (1, 2, 3, 4),
    (BlockWarpChannels<static_channels, interp_type>(sample, block, mapping, border)),
    (BlockWarpChannels<-1, interp_type>(sample, block, mapping, border)));
}

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_BLOCK_WARP_CUH_
