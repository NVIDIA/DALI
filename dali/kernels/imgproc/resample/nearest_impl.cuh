// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_

#include <cuda_runtime.h>
#ifndef __CUDACC__
#include <algorithm>
#endif
#include "dali/core/convert.h"
#include "dali/core/math_util.h"

namespace dali {
namespace kernels {

/**
 * @brief 2D nearest neighbor resampling
 *
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param origin - source coordinates corresponding to output (0, 0)
 * @param scale - step, in source coordinates, for one pixel in output coordinates
 * @param out_strides - stride between output rows
 * @param in_strides - stride between input rows
 * @param in_shape - shape of the input (x, y) order
 */
template <typename Dst, typename Src>
__device__ void NNResample(
    ivec2 lo, ivec2 hi,
    vec2 origin, vec2 scale,
    Dst *__restrict__ out,  vec<1, ptrdiff_t> out_stride,
    const Src *__restrict__ in, vec<1, ptrdiff_t> in_stride, ivec2 in_size, int channels) {
  origin += 0.5f * scale;
  for (int y = lo.y + threadIdx.y; y < hi.y; y += blockDim.y) {
    int ysrc = floor_int(__fmaf_rn(y, scale.y, origin.y));
    ysrc = clamp(ysrc, 0, in_size.y-1);

    Dst *out_row = &out[y * out_stride.x];
    const Src *in_row = &in[ysrc * in_stride.x];

    for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x) {
      int xsrc = floor_int(__fmaf_rn(x, scale.x, origin.x));
      xsrc = clamp(xsrc, 0, in_size.x-1);
      const Src *src_px = &in_row[xsrc * channels];
      for (int c = 0; c < channels; c++)
        out_row[x*channels + c] = ConvertSat<Dst>(__ldg(&src_px[c]));
    }
  }
}


/**
 * @brief 3D nearest neighbor resampling
 *
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 *
 * @param origin - source coordinates corresponding to output (0, 0, 0)
 * @param scale - step, in source coordinates, for one pixel in output coordinates
 */
template <typename Dst, typename Src>
__device__ void NNResample(
    ivec3 lo, ivec3 hi,
    vec3 origin, vec3 scale,
    Dst *__restrict__ out,  vec<2, ptrdiff_t> out_stride,
    const Src *__restrict__ in, vec<2, ptrdiff_t> in_stride, ivec3 in_size, int channels) {
  origin += 0.5f * scale;

  for (int z = lo.z + threadIdx.z; z < hi.z; z += blockDim.z) {
    int zsrc = floor_int(__fmaf_rn(z, scale.z, origin.z));
    zsrc = clamp(zsrc, 0, in_size.z-1);

    Dst *out_plane = &out[z * out_stride.y];
    const Src *in_plane = &in[zsrc * in_stride.y];

    for (int y = lo.y + threadIdx.y; y < hi.y; y += blockDim.y) {
      int ysrc = floor_int(__fmaf_rn(y, scale.y, origin.y));
      ysrc = clamp(ysrc, 0, in_size.y-1);

      Dst *out_row = &out_plane[y * out_stride.x];
      const Src *in_row = &in_plane[ysrc * in_stride.x];

      for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x) {
        int xsrc = floor_int(__fmaf_rn(x, scale.x, origin.x));
        xsrc = clamp(xsrc, 0, in_size.x-1);
        const Src *src_px = &in_row[xsrc * channels];
        for (int c = 0; c < channels; c++)
          out_row[x*channels + c] = ConvertSat<Dst>(__ldg(&src_px[c]));
      }
    }
  }
}


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_NEAREST_IMPL_CUH_
