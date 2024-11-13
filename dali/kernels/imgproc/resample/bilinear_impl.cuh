// Copyright (c) 2019, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_

#include <cuda_runtime.h>
#ifndef __CUDACC__
#include <algorithm>
#endif
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"
#include "dali/core/math_util.h"

namespace dali {
namespace kernels {

namespace {  // NOLINT

template <int n>
using ptrdiff_vec = vec<n, ptrdiff_t>;

/**
 * @brief Implements horizontal resampling, possibly with number of channels known at compile-time
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void LinearHorz_Channels(
    ivec2 lo, ivec2 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<1> in_strides, ivec2 in_shape, int dynamic_channels) {
  src_x0 += 0.5f * scale - 0.5f;

  ptrdiff_t out_stride = out_strides[0];
  ptrdiff_t in_stride = in_strides[0];
  int in_w = in_shape.x;

  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int x = lo.x + threadIdx.x; x < hi.x; x += blockDim.x) {
    const float sx0f = x * scale + src_x0;
    const int sx0i = __float2int_rd(sx0f);
    const float q = sx0f - sx0i;
    const int sx0 = clamp(sx0i, 0, in_w-1);
    const int sx1 = clamp(sx0i+1, 0, in_w-1);

    const Src *in_col1 = &in[sx0 * channels];
    const Src *in_col2 = &in[sx1 * channels];

    const int cx = channels * x;

    for (int y = threadIdx.y + lo.y; y < hi.y; y += blockDim.y) {
      Dst *out_row = &out[y * out_stride];
      const Src *in0 = &in_col1[y * in_stride];
      const Src *in1 = &in_col2[y * in_stride];

      for (int c = 0; c < channels; c++) {
        float a = __ldg(&in0[c]);
        float b = __ldg(&in1[c]);
        float tmp = __fmaf_rn(b-a, q, a);
        out_row[cx + c] = ConvertSat<Dst>(tmp);
      }
    }
  }
}

}  // namespace


/**
 * @brief Implements horizontal resampling
 *
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_x0 - X coordinate in the source image corresponding to output 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 *
 * The input region of interest is defined in terms of origin/scale, which are relative to
 * output (0, 0).
 * The lo/hi parameters are not output RoI - they merely indicate the output slice processed
 * by current block.
 */
template <int spatial_ndim, typename Dst, typename Src>
__device__ void LinearHorz(
    ivec<spatial_ndim> lo, ivec<spatial_ndim> hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<spatial_ndim-1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<spatial_ndim-1> in_strides,
    ivec<spatial_ndim> in_shape, int channels) {
  // Specialize over common numbers of channels.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    LinearHorz_Channels<static_channels>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      static_channels);
  ),  // NOLINT
  (
    LinearHorz_Channels<-1>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      channels);
  ));  // NOLINT
}

/**
 * @brief Implements vertical resampling
 *
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_y0 - Y coordinate in the source image corresponding to output 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 *
 * The input region of interest is defined in terms of origin/scale, which are relative to
 * output (0, 0).
 * The lo/hi parameters are not output RoI - they merely indicate the output slice processed
 * by current block.
 */
template <typename Dst, typename Src>
__device__ void LinearVert(
    ivec2 lo, ivec2 hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<1> in_strides, ivec2 in_shape, int channels) {
  src_y0 += __fmaf_rn(scale, 0.5f, -0.5f);

  ptrdiff_t out_stride = out_strides[0];
  ptrdiff_t in_stride = in_strides[0];
  int in_h = in_shape.y;

  // columns are independent - we can safely merge columns with channels
  const int j0 = lo.x * channels;
  const int j1 = hi.x * channels;

  for (int y = lo.y + threadIdx.y; y < hi.y; y += blockDim.y) {
    const float sy0f = __fmaf_rn(y, scale, src_y0);
    const int sy0i = __float2int_rd(sy0f);
    const float q = sy0f - sy0i;
    const int sy0 = clamp(sy0i,   0, in_h-1);
    const int sy1 = clamp(sy0i+1, 0, in_h-1);

    Dst *out_row = &out[y * out_stride];
    const Src *in0 = &in[sy0 * in_stride];
    const Src *in1 = &in[sy1 * in_stride];

    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
      float a = __ldg(&in0[j]);
      float b = __ldg(&in1[j]);
      float tmp = __fmaf_rn(b-a, q, a);
      out_row[j] = ConvertSat<Dst>(tmp);
    }
  }
}

template <typename Dst, typename Src>
__device__ void LinearDepth(
    ivec2 lo, ivec2 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<1> in_strides, ivec2 in_shape, int channels) {
  // Unreachable code - no assert to avoid excessive register pressure.
}


template <int static_channels, typename Dst, typename Src>
__device__ void LinearHorz_Channels(
    ivec3 lo, ivec3 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int channels) {
  for (int z = lo.z + threadIdx.z; z < hi.z; z += blockDim.z) {
    ptrdiff_t out_ofs = z * out_strides.y;
    ptrdiff_t in_ofs  = z * in_strides.y;
    LinearHorz_Channels<static_channels>(
              sub<2>(lo), sub<2>(hi), src_x0, scale,
              out + out_ofs, sub<1>(out_strides),
              in + in_ofs, sub<1>(in_strides), sub<2>(in_shape), channels);
  }
}

template <typename Dst, typename Src>
__device__ void LinearHorz(
    ivec3 lo, ivec3 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int channels) {
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    LinearHorz_Channels<static_channels>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      static_channels);
  ),  // NOLINT
  (
    LinearHorz_Channels<-1>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      channels);
  ));  // NOLINT
}

template <typename Dst, typename Src>
__device__ void LinearVert(
    ivec3 lo, ivec3 hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int channels) {

  for (int z = lo.z + threadIdx.z; z < hi.z; z += blockDim.z) {
    ptrdiff_t out_ofs = z * out_strides.y;
    ptrdiff_t in_ofs  = z * in_strides.y;
    LinearVert(sub<2>(lo), sub<2>(hi), src_y0, scale,
               out + out_ofs, sub<1>(out_strides),
               in + in_ofs, sub<1>(in_strides), sub<2>(in_shape), channels);
  }
}


/**
 * @brief Implements depthwise resampling
 *
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_z0 - Z coordinate in the source image corresponding to output 0
 * @param scale - step, in source Z, for one pixel in output Z (may be negative)
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 *
 * The input region of interest is defined in terms of origin/scale, which are relative to
 * output (0, 0).
 * The lo/hi parameters are not output RoI - they merely indicate the output slice processed
 * by current block.
 */
template <typename Dst, typename Src>
__device__ void LinearDepth(
    ivec3 lo, ivec3 hi,
    float src_z0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int channels) {
  src_z0 += 0.5f * scale - 0.5f;

  // columns are independent - we can safely merge columns with channels
  const int j0 = lo.x * channels;
  const int j1 = hi.x * channels;

  ptrdiff_t out_stride_y = out_strides[0];
  ptrdiff_t out_stride_z = out_strides[1];
  ptrdiff_t in_stride_y = in_strides[0];
  ptrdiff_t in_stride_z = in_strides[1];


  // threadIdx.y is used to traverse Z axis
  for (int z = lo.z + threadIdx.y; z < hi.z; z += blockDim.y) {
    const float sz0f = z * scale + src_z0;
    const int sz0i = __float2int_rd(sz0f);
    const float q = sz0f - sz0i;
    const int sz0 = clamp(sz0i,   0, in_shape.z-1);
    const int sz1 = clamp(sz0i+1, 0, in_shape.z-1);

    Dst *out_slice = &out[z * out_stride_z];
    const Src *in_slice0 = &in[sz0 * in_stride_z];
    const Src *in_slice1 = &in[sz1 * in_stride_z];

    // cannot fuse X and Y due to RoI support
    for (int y = lo.y + threadIdx.z; y < hi.y; y += blockDim.z) {
      Dst *out_row = &out_slice[y * out_stride_y];
      const Src *in0 = &in_slice0[y * in_stride_y];
      const Src *in1 = &in_slice1[y * in_stride_y];

      for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        float a = __ldg(&in0[j]);
        float b = __ldg(&in1[j]);
        float tmp = __fmaf_rn(b-a, q, a);
        out_row[j] = ConvertSat<Dst>(tmp);
      }
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_
