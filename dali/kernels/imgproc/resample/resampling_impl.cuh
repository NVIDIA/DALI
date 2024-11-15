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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/core/geom/vec.h"
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"

namespace dali {
namespace kernels {

namespace {  // NOLINT

template <int n>
using ptrdiff_vec = vec<n, ptrdiff_t>;

namespace resample_shared {
  extern __shared__ float coeffs[];
};

/**
 * @brief Implements horizontal resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_x0 - X coordinate in the source image corresponding to output 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param support - size of the resampling kernel, in source pixels
 * @tparam static_channels - number of channels, if known at compile time
 *
 * The function fills the output in block-sized vertical spans.
 * Block horizontal size is warp-aligned.
 * Filter coefficients are pre-calculated for each vertical span to avoid
 * recalculating them for each row, and stored in a shared memory block.
 *
 * The function follows different code paths for static and dynamic number of channels.
 * For the dynamic, the innermost loop goes over filter taps, which eliminates the need
 * for thread-local memory to store intermediate sums. This allows processing arbitrary
 * number of channels.
 * For static number of channels, the run-time parameter `channels` is ignored and
 * there's also a local temporary storage for a tap sum for each channel. This is faster,
 * but requires extra registers for the intermediate sums.
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleHorz_Channels(
    ivec2 lo, ivec2 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<1> in_strides, ivec2 in_shape, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  ptrdiff_t out_stride = out_strides.x;
  ptrdiff_t in_stride = in_strides.x;
  int in_w = in_shape.x;

  src_x0 += __fmaf_rn(scale, 0.5f, -0.5f) - filter.anchor;

  const float filter_step = filter.scale;


  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : threadIdx.x;
  const int coeff_stride = huge_kernel ? 1 : blockDim.x;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int j = lo.x; j < hi.x; j += blockDim.x) {
    int dx = j + threadIdx.x;
    const float sx0f = __fmaf_rn(dx, scale, src_x0);
    const int sx0 = huge_kernel ? __float2int_rn(sx0f) : __float2int_ru(sx0f);
    float f = (sx0 - sx0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.y; k < support; k += blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[coeff_base + coeff_stride*k] = flt;
      }
    }
    __syncthreads();

    if (dx >= hi.x)
      continue;

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + coeff_stride * k];
    }
    norm = 1.0f / norm;

    for (int i = threadIdx.y + lo.y; i < hi.y; i += blockDim.y) {
      const Src *in_row = &in[i * in_stride];
      Dst *out_row = &out[i * out_stride];

      if (static_channels < 0) {
        for (int c = 0; c < channels; c++) {
          float tmp = 0;

          for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
            int x = sx0 + k;
            int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
            float flt = coeffs[coeff_idx];
            Src px = __ldg(in_row + channels * xsample + c);
            tmp = __fmaf_rn(px, flt, tmp);
          }

          out_row[channels * dx + c] = ConvertSat<Dst>(tmp * norm);
        }

      } else {
        float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
        for (int c = 0; c < channels; c++)
          tmp[c] = 0;

        for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
          int x = sx0 + k;
          int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
          float flt = coeffs[coeff_idx];
          for (int c = 0; c < channels; c++) {
            Src px = __ldg(in_row + channels * xsample + c);
            tmp[c] = __fmaf_rn(px, flt, tmp[c]);
          }
        }

        for (int c = 0; c < channels; c++)
          out_row[channels * dx + c] = ConvertSat<Dst>(tmp[c] * norm);
      }
    }
  }
}



/**
 * @brief Implements horizontal resampling for a custom ROI
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_x0 - X coordinate in the source image corresponding to output 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param support - size of the resampling kernel, in source voxels
 * @tparam static_channels - number of channels, if known at compile time
 *
 * The function fills the output in block-sized vertical spans.
 * Block horizontal size is warp-aligned.
 * Filter coefficients are pre-calculated for each vertical span to avoid
 * recalculating them for each row, and stored in a shared memory block.
 *
 * The function follows different code paths for static and dynamic number of channels.
 * For the dynamic, the innermost loop goes over filter taps, which eliminates the need
 * for thread-local memory to store intermediate sums. This allows processing arbitrary
 * number of channels.
 * For static number of channels, the run-time parameter `channels` is ignored and
 * there's also a local temporary storage for a tap sum for each channel. This is faster,
 * but requires extra registers for the intermediate sums.
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleHorz_Channels(
    ivec3 lo, ivec3 hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  ptrdiff_t out_stride_y = out_strides.x;  // coordinates are shifted, because
  ptrdiff_t out_stride_z = out_strides.y;  // X stride is implicitly equal to the number of channels
  ptrdiff_t in_stride_y = in_strides.x;
  ptrdiff_t in_stride_z = in_strides.y;
  int in_w = in_shape.x;

  src_x0 += __fmaf_rn(scale, 0.5f, -0.5f) - filter.anchor;

  const float filter_step = filter.scale;


  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : threadIdx.x;
  const int coeff_stride = huge_kernel ? 1 : blockDim.x;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int j = lo.x; j < hi.x; j += blockDim.x) {
    int dx = j + threadIdx.x;
    const float sx0f = __fmaf_rn(dx, scale, src_x0);
    const int sx0 = huge_kernel ? __float2int_rn(sx0f) : __float2int_ru(sx0f);
    float f = (sx0 - sx0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.y; k < support; k += blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[coeff_base + coeff_stride*k] = flt;
      }
    }
    __syncthreads();

    if (dx >= hi.x)
      continue;

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + coeff_stride * k];
    }
    norm = 1.0f / norm;

    for (int z = threadIdx.z + lo.z; z < hi.z; z += blockDim.z) {
      const Src *in_slice = &in[z * in_stride_z];
      Dst *out_slice = &out[z * out_stride_z];

      for (int i = threadIdx.y + lo.y; i < hi.y; i += blockDim.y) {
        const Src *in_row = &in_slice[i * in_stride_y];
        Dst *out_row = &out_slice[i * out_stride_y];

        if (static_channels < 0) {
          for (int c = 0; c < channels; c++) {
            float tmp = 0;

            for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
              int x = sx0 + k;
              int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
              float flt = coeffs[coeff_idx];
              Src px = __ldg(in_row + channels * xsample + c);
              tmp = __fmaf_rn(px, flt, tmp);
            }

            out_row[channels * dx + c] = ConvertSat<Dst>(tmp * norm);
          }

        } else {
          float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
          for (int c = 0; c < channels; c++)
            tmp[c] = 0;

          for (int k = 0, coeff_idx=coeff_base; k < support; k++, coeff_idx += coeff_stride) {
            int x = sx0 + k;
            int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
            float flt = coeffs[coeff_idx];
            for (int c = 0; c < channels; c++) {
              Src px = __ldg(in_row + channels * xsample + c);
              tmp[c] = __fmaf_rn(px, flt, tmp[c]);
            }
          }

          for (int c = 0; c < channels; c++)
            out_row[channels * dx + c] = ConvertSat<Dst>(tmp[c] * norm);
        }
      }
    }
  }
}


/**
 * @brief Implements vertical resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_y0 - Y coordinate in the source image corresponding to output 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param support - size of the resampling kernel, in source pixels
 * @tparam static_channels - number of channels, if known at compile time
 *
 * The function fills the output in block-sized horizontal spans.
 * Filter coefficients are pre-calculated for each horizontal span to avoid
 * recalculating them for each column, and stored in a shared memory block.
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleVert_Channels(
    ivec2 lo, ivec2 hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<1> in_strides, ivec2 in_shape, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  ptrdiff_t out_stride = out_strides.x;
  ptrdiff_t in_stride = in_strides.x;
  int in_h = in_shape.y;

  src_y0 += __fmaf_rn(scale, 0.5f, -0.5f) - filter.anchor;

  const float filter_step = filter.scale;

  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : support*threadIdx.y;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int i = lo.y; i < hi.y; i+=blockDim.y) {
    int dy = i + threadIdx.y;
    const float sy0f = __fmaf_rn(dy, scale, src_y0);
    const int sy0 = huge_kernel ? __float2int_rn(sy0f) : __float2int_ru(sy0f);
    float f = (sy0 - sy0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.x; k < support; k += blockDim.x) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[coeff_base + k] = flt;
      }
    }
    __syncthreads();

    if (dy >= hi.y)
      continue;

    Dst *out_row = &out[dy * out_stride];

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    for (int j = lo.x + threadIdx.x; j < hi.x; j += blockDim.x) {
      Dst *out_col = &out_row[j * channels];
      const Src *in_col = &in[j * channels];

      if (static_channels < 0) {
        for (int c = 0; c < channels; c++) {
          float tmp = 0;

          for (int k = 0; k < support; k++) {
            int y = sy0 + k;
            int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
            float flt = coeffs[coeff_base + k];
            Src px = __ldg(in_col + in_stride * ysample + c);
            tmp = __fmaf_rn(px, flt, tmp);
          }

          out_col[c] = ConvertSat<Dst>(tmp * norm);
        }
      } else {
        float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
        for (int c = 0; c < channels; c++)
          tmp[c] = 0;

        for (int k = 0; k < support; k++) {
          int y = sy0 + k;
          int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
          float flt = coeffs[coeff_base + k];
          for (int c = 0; c < channels; c++) {
            Src px = __ldg(in_col + in_stride * ysample + c);
            tmp[c] = __fmaf_rn(px, flt, tmp[c]);
          }
        }

        for (int c = 0; c < channels; c++)
          out_col[c] = ConvertSat<Dst>(tmp[c] * norm);
      }
    }
  }
}

/**
 * @brief Implements vertical resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_y0 - Y coordinate in the source image corresponding to output 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param support - size of the resampling kernel, in source voxels
 * @tparam static_channels - number of channels, if known at compile time
 *
 * The function fills the output in block-sized horizontal/depthwise spans.
 * Filter coefficients are pre-calculated for each span to avoid recalculating them for each
 * column, and stored in a shared memory block.
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleVert_Channels(
    ivec3 lo, ivec3 hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  ptrdiff_t out_stride_y = out_strides.x;
  ptrdiff_t in_stride_y = in_strides.x;
  ptrdiff_t out_stride_z = out_strides.y;
  ptrdiff_t in_stride_z = in_strides.y;
  int in_h = in_shape.y;

  src_y0 += __fmaf_rn(scale, 0.5f, -0.5f) - filter.anchor;

  const float filter_step = filter.scale;

  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : support*threadIdx.y;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  for (int i = lo.y; i < hi.y; i+=blockDim.y) {
    int dy = i + threadIdx.y;
    const float sy0f = __fmaf_rn(dy, scale, src_y0);
    const int sy0 = huge_kernel ? __float2int_rn(sy0f) : __float2int_ru(sy0f);
    float f = (sy0 - sy0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.x; k < support; k += blockDim.x) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[coeff_base + k] = flt;
      }
    }
    __syncthreads();

    if (dy >= hi.y)
      continue;

    Dst *out_row = &out[dy * out_stride_y];

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    for (int z = lo.z + threadIdx.z; z < hi.z; z += blockDim.z) {
      Dst *out_slice = &out_row[z * out_stride_z];
      const Src *in_slice = &in[z * in_stride_z];
      for (int j = lo.x + threadIdx.x; j < hi.x; j += blockDim.x) {
        Dst *out_col = &out_slice[j * channels];
        const Src *in_col = &in_slice[j * channels];

        if (static_channels < 0) {
          for (int c = 0; c < channels; c++) {
            float tmp = 0;

            for (int k = 0; k < support; k++) {
              int y = sy0 + k;
              int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
              float flt = coeffs[coeff_base + k];
              Src px = __ldg(in_col + in_stride_y * ysample + c);
              tmp = __fmaf_rn(px, flt, tmp);
            }

            out_col[c] = ConvertSat<Dst>(tmp * norm);
          }
        } else {
          float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
          for (int c = 0; c < channels; c++)
            tmp[c] = 0;

          for (int k = 0; k < support; k++) {
            int y = sy0 + k;
            int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
            float flt = coeffs[coeff_base + k];
            for (int c = 0; c < channels; c++) {
              Src px = __ldg(in_col + in_stride_y * ysample + c);
              tmp[c] = __fmaf_rn(px, flt, tmp[c]);
            }
          }

          for (int c = 0; c < channels; c++)
            out_col[c] = ConvertSat<Dst>(tmp[c] * norm);
        }
      }
    }
  }
}

template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleDepth_Channels(
    ivec2 /*lo*/, ivec2 /*hi*/,
    float /*src_z0*/, float /*scale*/,
    Dst *__restrict__ /*out*/, ptrdiff_vec<1> /*out_strides*/,
    const Src *__restrict__ /*in*/, ptrdiff_vec<1> /*in_strides*/, ivec2 /*in_shape*/,
    int /*dynamic_channels*/,
    ResamplingFilter /*filter*/, int /*support*/) {
  // Unreachable code - no assert to avoid excessive register pressure.
}

/**
 * @brief Implements depthwise resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_z0 - start source coordinate in Z axis
 * @param scale - dest-to-source scale in Z axis
 * @param support - size of the resampling kernel, in source voxels
 * @tparam static_channels - number of channels, if known at compile time
 */
template <int static_channels = -1, typename Dst, typename Src>
__device__ void ResampleDepth_Channels(
    ivec3 lo, ivec3 hi,
    float src_z0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<2> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<2> in_strides, ivec3 in_shape, int dynamic_channels,
    ResamplingFilter filter, int support) {
  using resample_shared::coeffs;

  ptrdiff_t out_stride_y = out_strides[0];
  ptrdiff_t out_stride_z = out_strides[1];
  ptrdiff_t in_stride_y = in_strides[0];
  ptrdiff_t in_stride_z = in_strides[1];
  int in_d = in_shape.z;

  src_z0 += __fmaf_rn(scale, 0.5f, -0.5f) - filter.anchor;

  const float filter_step = filter.scale;

  const bool huge_kernel = support > 256;
  const int coeff_base = huge_kernel ? 0 : support*threadIdx.y;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  // threadIdx.y is used to traverse Z axis
  for (int i = lo.z; i < hi.z; i+=blockDim.y) {
    int dz = i + threadIdx.y;
    const float sz0f = __fmaf_rn(dz, scale, src_z0);
    const int sz0 = huge_kernel ? __float2int_rn(sz0f) : __float2int_ru(sz0f);
    float f = (sz0 - sz0f) * filter_step;
    __syncthreads();
    if (huge_kernel) {
      for (int k = threadIdx.x + blockDim.x*threadIdx.y; k < support; k += blockDim.x*blockDim.y) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[k] = flt;
      }
    } else {
      for (int k = threadIdx.x; k < support; k += blockDim.x) {
        float flt = filter(__fmaf_rn(k, filter_step, f));
        coeffs[coeff_base + k] = flt;
      }
    }
    __syncthreads();

    if (dz >= hi.z)
      continue;

    Dst *out_slice = &out[dz * out_stride_z];

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    // cannot fuse X and Y due to RoI support
    for (int j = lo.y + threadIdx.z; j < hi.y; j += blockDim.z) {
      Dst *out_row = &out_slice[j * out_stride_y];
      const Src *in_row = &in[j * in_stride_y];
      for (int k = lo.x + threadIdx.x; k < hi.x; k += blockDim.x) {
        Dst *out_col = &out_row[k * channels];
        const Src *in_col = &in_row[k * channels];

        if (static_channels < 0) {
          for (int c = 0; c < channels; c++) {
            float tmp = 0;

            for (int l = 0; l < support; l++) {
              int z = sz0 + l;
              int zsample = z < 0 ? 0 : z >= in_d-1 ? in_d-1 : z;
              float flt = coeffs[coeff_base + l];
              Src px = __ldg(in_col + in_stride_z * zsample + c);
              tmp = __fmaf_rn(px, flt, tmp);
            }

            out_col[c] = ConvertSat<Dst>(tmp * norm);
          }
        } else {
          float tmp[static_channels < 0 ? 1 : static_channels];  // NOLINT - not a variable length array
          for (int c = 0; c < channels; c++)
            tmp[c] = 0;

          for (int l = 0; l < support; l++) {
            int z = sz0 + l;
            int zsample = z < 0 ? 0 : z >= in_d-1 ? in_d-1 : z;
            float flt = coeffs[coeff_base + l];
            for (int c = 0; c < channels; c++) {
              Src px = __ldg(in_col + in_stride_z * zsample + c);
              tmp[c] = __fmaf_rn(px, flt, tmp[c]);
            }
          }

          for (int c = 0; c < channels; c++)
            out_col[c] = ConvertSat<Dst>(tmp[c] * norm);
        }
      }
    }
  }
}


}  // namespace

/**
 * @brief Implements horizontal resampling for a custom ROI
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_x0 - X coordinate in the source image corresponding to output 0
 * @param scale - step, in source X, for one pixel in output X (may be negative)
 * @param support - size of the resampling kernel, in source samples
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 */
template <int spatial_ndim, typename Dst, typename Src>
__device__ void ResampleHorz(
    ivec<spatial_ndim> lo, ivec<spatial_ndim> hi,
    float src_x0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<spatial_ndim-1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<spatial_ndim-1> in_strides,
    ivec<spatial_ndim> in_shape, int channels,
    ResamplingFilter filter, int support) {
  // Specialize over common numbers of channels.
  // Ca. 20% speedup compared to generic code path for
  // three channel image with large kernel.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    ResampleHorz_Channels<static_channels>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      static_channels, filter, support);
  ),  // NOLINT
  (
    ResampleHorz_Channels<-1>(
      lo, hi, src_x0, scale,
      out, out_strides, in, in_strides, in_shape,
      channels, filter, support);
  ));  // NOLINT
}

/**
 * @brief Implements vertical resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_y0 - Y coordinate in the source image corresponding to output 0
 * @param scale - step, in source Y, for one pixel in output Y (may be negative)
 * @param support - size of the resampling kernel, in source samples
 * @param out_strides - stride between output rows (and slices)
 * @param in_strides - stride between input rows (and slices)
 * @param in_shape - shape of the input (x, y[, z]) order
 */
template <int spatial_ndim, typename Dst, typename Src>
__device__ void ResampleVert(
    ivec<spatial_ndim> lo, ivec<spatial_ndim> hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<spatial_ndim-1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<spatial_ndim-1> in_strides,
    ivec<spatial_ndim> in_shape, int channels,
    ResamplingFilter filter, int support) {
  // Specialize over common numbers of channels.
  // Ca. 20% speedup compared to generic code path for
  // three channel image with large kernel.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    ResampleVert_Channels<static_channels>(
      lo, hi, src_y0, scale,
      out, out_strides, in, in_strides, in_shape,
      static_channels, filter, support);
  ),  // NOLINT
  (
    ResampleVert_Channels<-1>(
      lo, hi, src_y0, scale,
      out, out_strides, in, in_strides, in_shape,
      channels, filter, support);
  ));  // NOLINT
}

/**
 * @brief Implements depthwise resampling
 * @param lo - inclusive lower bound output coordinates
 * @param hi - exclusive upper bound output coordinates
 * @param src_z0 - start source coordinate in Z axis
 * @param scale - dest-to-source scale in Z axis
 * @param support - size of the resampling kernel, in source samples
 * @param out_strides - stride between output rows and slices
 * @param in_strides - stride between input rows and slices
 * @param in_shape - shape of the input (x, y, z)
 */
template <int spatial_ndim, typename Dst, typename Src>
__device__ void ResampleDepth(
    ivec<spatial_ndim> lo, ivec<spatial_ndim> hi,
    float src_y0, float scale,
    Dst *__restrict__ out, ptrdiff_vec<spatial_ndim-1> out_strides,
    const Src *__restrict__ in, ptrdiff_vec<spatial_ndim-1> in_strides,
    ivec<spatial_ndim> in_shape, int channels,
    ResamplingFilter filter, int support) {
  // Specialize over common numbers of channels.
  // Ca. 20% speedup compared to generic code path for
  // three channel image with large kernel.
  VALUE_SWITCH(channels, static_channels, (1, 2, 3, 4),
  (
    ResampleDepth_Channels<static_channels>(
      lo, hi, src_y0, scale,
      out, out_strides, in, in_strides, in_shape,
      static_channels, filter, support);
  ),  // NOLINT
  (
    ResampleDepth_Channels<-1>(
      lo, hi, src_y0, scale,
      out, out_strides, in, in_strides, in_shape,
      channels, filter, support);
  ));  // NOLINT
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
