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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/kernels/static_switch.h"
#include "dali/kernels/common/convert.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"

namespace dali {
namespace kernels {

constexpr int ResampleSharedMemSize = 32<<10;

namespace {

namespace resample_shared {
  extern __shared__ float coeffs[];
};

/// @brief Implements horizontal resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
/// @tparam static_channels - number of channels, if known at compile time
///
/// The function fills the output in block-sized vertical spans.
/// Block horizontal size is warp-aligned.
/// Filter coefficients are pre-calculated for each vertical span to avoid
/// recalculating them for each row, and stored in a shared memory block.
///
/// The function follows different code paths for static and dynamic number of channels.
/// For the dynamic, the innermost loop goes over filter taps, which eliminates the need
/// for thread-local memory to store intermediate sums. This allows processing arbitrary
/// number of channels.
/// For static number of channels, the run-time parameter `channels` is ignored and
/// there's also a local temporary storage for a tap sum for each channel. This is faster,
/// but requires extra registers for the intermediate sums.
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleHorz_Channels(
    int x0, int x1, int y0, int y1,
    float src_x0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_w, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  src_x0 += 0.5f * scale - 0.5f - filter.anchor;

  const float filter_step = filter.scale;


  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : threadIdx.x;
  const int coeff_stride = huge_kernel ? 1 : blockDim.x;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  const float bias = std::is_integral<Dst>::value ? 0.5f : 0;

  for (int j = x0; j < x1; j += blockDim.x) {
    int dx = j + threadIdx.x;
    const float sx0f = dx * scale + src_x0;
    const int sx0 = huge_kernel ? __float2int_rn(sx0f) : __float2int_ru(sx0f);
    float f = (sx0 - sx0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(f + k*filter_step);
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.y; k < support; k += blockDim.y) {
        float flt = filter(f + k*filter_step);
        coeffs[coeff_base + coeff_stride*k] = flt;
      }
    }
    __syncthreads();

    if (dx >= x1)
      continue;

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + coeff_stride * k];
    }
    norm = 1.0f / norm;

    for (int i = threadIdx.y + y0; i < y1; i+=blockDim.y) {
      const Src *in_row = &in[i * in_stride];
      Dst *out_row = &out[i * out_stride];

      if (static_channels < 0) {
        for (int c = 0; c < channels; c++) {
          float tmp = bias;

          for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
            int x = sx0 + k;
            int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
            float flt = coeffs[coeff_idx];
            Src px = __ldg(in_row + channels * xsample + c);
            tmp += px * flt;
          }

          out_row[channels * dx + c] = clamp<Dst>(tmp * norm);
        }

      } else {
        float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
        for (int c = 0; c < channels; c++)
          tmp[c] = bias;

        for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
          int x = sx0 + k;
          int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
          float flt = coeffs[coeff_idx];
          for (int c = 0; c < channels; c++) {
            Src px = __ldg(in_row + channels * xsample + c);
            tmp[c] += px * flt;
          }
        }

        for (int c = 0; c < channels; c++)
          out_row[channels * dx + c] = clamp<Dst>(tmp[c] * norm);
      }
    }
  }
}

/// @brief Implements vertical resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
/// @tparam static_channels - number of channels, if known at compile time
///
/// The function fills the output in block-sized horizontal spans.
/// Filter coefficients are pre-calculated for each horizontal span to avoid
/// recalculating them for each column, and stored in a shared memory block.
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleVert_Channels(
    int x0, int x1, int y0, int y1,
    float src_y0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_h, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  src_y0 += 0.5f * scale - 0.5f - filter.anchor;

  const float filter_step = filter.scale;

  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : support*threadIdx.y;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  const float bias = std::is_integral<Dst>::value ? 0.5f : 0;

  for (int i = y0; i < y1; i+=blockDim.y) {
    int dy = i + threadIdx.y;
    const float sy0f = dy * scale + src_y0;
    const int sy0 = huge_kernel ? __float2int_rn(sy0f) : __float2int_ru(sy0f);
    float f = (sy0 - sy0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(f + k*filter_step);
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.x; k < support; k += blockDim.x) {
        float flt = filter(f + k*filter_step);
        coeffs[coeff_base + k] = flt;
      }
    }
    __syncthreads();

    if (dy >= y1)
      continue;

    Dst *out_row = &out[dy * out_stride];

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    for (int j = x0 + threadIdx.x; j < x1; j += blockDim.x) {
      Dst *out_col = &out_row[j * channels];
      const Src *in_col = &in[j * channels];

      if (static_channels < 0) {
        for (int c = 0; c < channels; c++) {
          float tmp = bias;

          for (int k = 0; k < support; k++) {
            int y = sy0 + k;
            int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
            float flt = coeffs[coeff_base + k];
            Src px = __ldg(in_col + in_stride * ysample + c);
            tmp += px * flt;
          }

          out_col[c] = clamp<Dst>(tmp * norm);
        }
      } else {
        float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
        for (int c = 0; c < channels; c++)
          tmp[c] = bias;

        for (int k = 0; k < support; k++) {
          int y = sy0 + k;
          int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
          float flt = coeffs[coeff_base + k];
          for (int c = 0; c < channels; c++) {
            Src px = __ldg(in_col + in_stride * ysample + c);
            tmp[c] += px * flt;
          }
        }

        for (int c = 0; c < channels; c++)
          out_col[c] = clamp<Dst>(tmp[c] * norm);
      }
    }
  }
}

}  // namespace

template <typename Dst, typename Src>
__device__ void ResampleHorz(
    int x0, int x1, int y0, int y1,
    float src_x0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_w, int channels,
    ResamplingFilter filter, int support) {
  // Specialize over common numbers of channels.
  // Ca. 20% speedup compared to generic code path for
  // three channel image with large kernel.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    ResampleHorz_Channels<static_channels>(
      x0, x1, y0, y1, src_x0, scale,
      out, out_stride, in, in_stride, in_w,
      static_channels, filter, support);
  ),  // NOLINT
  (
    ResampleHorz_Channels<-1>(
      x0, x1, y0, y1, src_x0, scale,
      out, out_stride, in, in_stride, in_w,
      channels, filter, support);
  ));  // NOLINT
}

template <typename Dst, typename Src>
__device__ void ResampleVert(
    int x0, int x1, int y0, int y1,
    float src_y0, float scale,
    Dst *__restrict__ out, int out_stride,
    const Src *__restrict__ in, int in_stride, int in_h, int channels,
    ResamplingFilter filter, int support) {
  // Specialize over common numbers of channels.
  // Ca. 20% speedup compared to generic code path for
  // three channel image with large kernel.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    ResampleVert_Channels<static_channels>(
      x0, x1, y0, y1, src_y0, scale,
      out, out_stride, in, in_stride, in_h,
      static_channels, filter, support);
  ),  // NOLINT
  (
    ResampleVert_Channels<-1>(
      x0, x1, y0, y1, src_y0, scale,
      out, out_stride, in, in_stride, in_h,
      channels, filter, support);
  ));  // NOLINT
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
