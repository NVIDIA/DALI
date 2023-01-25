// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_IMPL_CUH_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace slice_flip_normalize {

template <typename Out, typename In, int spatial_ndim>
struct SampleDesc {
  Surface2D<Out> out;
  Surface2D<const In> in;
  Roi<spatial_ndim> bounds;

  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  const void *__restrict__ fill_values;

  i64vec<spatial_ndim> out_strides;
  int64_t out_channel_stride;
  i64vec<spatial_ndim> in_strides;
  int64_t in_channel_stride;
};


template <typename Out, typename In>
__global__ void SliceNormalizeKernel_2D(const SampleDesc<Out, In, 2> *samples,
                                        const ::dali::kernels::BlockDesc<2> *tiles) {
  const auto tile = tiles[blockIdx.x];
  const auto sample = samples[tile.sample_idx];
  auto fill_values = static_cast<const Out *>(sample.fill_values);
  for (int y = threadIdx.y + tile.start.y; y < tile.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + tile.start.x; x < tile.end.x; x += blockDim.x) {
      int c = 0;
      if (!sample.bounds.contains(ivec2{x, y})) {
        for (; c < sample.out.channels; c++) {
          sample.out(x, y, c) = fill_values[c];
        }
      } else {
        for (; c < sample.in.channels; c++) {
          float fpin = sample.in(x, y, c);
          float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
          sample.out(x, y, c) = ConvertSat<Out>(fpout);
        }
        for (; c < sample.out.channels; c++) {
          sample.out(x, y, c) = fill_values[c];
        }
      }
    }
  }
}

template <int static_channels, typename Out, typename In>
__device__ void SliceNormalizeKernel_2D_NoPad_Ch(const SampleDesc<Out, In, 2> &sample,
                                              const ::dali::kernels::BlockDesc<2> &tile) {
  auto fill_values = static_cast<const Out *>(sample.fill_values);
  for (int y = threadIdx.y + tile.start.y; y < tile.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + tile.start.x; x < tile.end.x; x += blockDim.x) {
      if constexpr (static_channels > 0) {
        #pragma unroll static_channels
        for (int c = 0; c < static_channels; c++) {
          float fpin = sample.in(x, y, c);
          float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
          sample.out(x, y, c) = ConvertSat<Out>(fpout);
        }
      } else {
        for (int c = 0; c < sample.in.channels; c++) {
          float fpin = sample.in(x, y, c);
          float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
          sample.out(x, y, c) = ConvertSat<Out>(fpout);
        }
      }
    }
  }
}

template <typename Out, typename In>
__global__ void SliceNormalizeKernel_2D_NoPad(const SampleDesc<Out, In, 2> *samples,
                                              const ::dali::kernels::BlockDesc<2> *tiles) {
  const auto tile = tiles[blockIdx.x];
  const auto sample = samples[tile.sample_idx];
  VALUE_SWITCH(sample.out.channels, static_channels, (1, 2, 3, 4, 5, 6, 7, 8, 16),
    (SliceNormalizeKernel_2D_NoPad_Ch<static_channels>(sample, tile);),
    (SliceNormalizeKernel_2D_NoPad_Ch<-1>(sample, tile);)
  );  // NOLINT(whitespace/parens)
}


}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_IMPL_CUH_
