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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_

#include <cuda_runtime.h>
#ifndef __CUDACC__
#include <algorithm>
#endif
#include "dali/kernels/static_switch.h"
#include "dali/kernels/common/convert.h"

namespace dali {
namespace kernels {

namespace {

/// @brief Implements horizontal resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
template <int static_channels = -1, typename Dst, typename Src>
__device__ void LinearHorz_Channels(
    int x0, int x1, int y0, int y1,
    float src_x0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_w, int dynamic_channels) {
  src_x0 += 0.5f * scale - 0.5f;

  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int j = x0 + threadIdx.x; j < x1; j += blockDim.x) {
    const float sx0f = j * scale + src_x0;
    const int sx0 = min(max(0, static_cast<int>(floorf(sx0f))), in_w-1);
    const int sx1 = min(sx0+1, in_w-1);
    const float q = sx0f - sx0;
    const float p = 1-q;

    const Src *in_col1 = &in[sx0 * channels];
    const Src *in_col2 = &in[sx1 * channels];

    for (int i = threadIdx.y + y0; i < y1; i += blockDim.y) {
      Dst *out_row = &out[i * out_stride];
      const Src *in1 = &in_col1[i * in_stride];
      const Src *in2 = &in_col2[i * in_stride];

      for (int c = 0; c < channels; c++) {
        float tmp = __ldg(&in1[c]) * p + __ldg(&in2[c]) * q;
        if (std::is_integral<Dst>::value)
          tmp += 0.5f;
        out_row[channels * j + c] = clamp<Dst>(tmp);
      }
    }
  }
}

}  // namespace

template <typename Dst, typename Src>
__device__ void LinearHorz(
    int x0, int x1, int y0, int y1,
    float src_x0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_w, int channels) {
  // Specialize over common numbers of channels.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    LinearHorz_Channels<static_channels>(
      x0, x1, y0, y1, 0, scale,
      out, out_stride, in, in_stride, in_w,
      static_channels);
  ),  // NOLINT
  (
    LinearHorz_Channels<-1>(
      x0, x1, y0, y1, 0, scale,
      out, out_stride, in, in_stride, in_w,
      channels);
  ));  // NOLINT
}

/// @brief Implements vertical resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
template <typename Dst, typename Src>
__device__ void LinearVert(
    int x0, int x1, int y0, int y1,
    float src_y0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_h, int channels) {
  src_y0 += 0.5f * scale - 0.5f;

  // columns are independent - we can safely merge columns with channels
  const int j0 = x0 * channels;
  const int j1 = x1 * channels;

  for (int i = y0 + threadIdx.y; i < y1; i += blockDim.y) {
    const float sy0f = i * scale + src_y0;
    const int sy0 = min(max(0, static_cast<int>(floorf(sy0f))), in_h-1);
    const int sy1 = min(sy0+1, in_h-1);
    const float q = sy0f - sy0;
    const float p = 1-q;

    Dst *out_row = &out[i * out_stride];
    const Src *in1 = &in[sy0 * in_stride];
    const Src *in2 = &in[sy1 * in_stride];

    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
      float tmp = __ldg(&in1[j]) * p + __ldg(&in2[j]) * q;
      if (std::is_integral<Dst>::value)
        tmp += 0.5f;
      out_row[j] = clamp<Dst>(tmp);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_BILINEAR_IMPL_CUH_
