// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_
#define DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_

#include <cuda_runtime_api.h>
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/jpeg/dct_8x8_gpu.cuh"
#include "dali/core/geom/vec.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {

// Quantization table coefficients that are suggested in the Annex of the JPEG standard.
__constant__ uint8_t Q_luma[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

__constant__ uint8_t Q_chroma[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
};

struct SampleDesc {
  const uint8_t *in;  // rgb
  uint8_t *out;  // rgb
  ivec<2> size;
  i64vec<2> strides;
};

template <typename T>
__inline__ __device__ T rgb_to_y(vec<3, T> rgb) {
  return ConvertSat<T>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
}

template <typename T>
__inline__ __device__ T rgb_to_cb(vec<3, T> rgb) {
  return ConvertSat<T>(-0.16873589f * rgb.x - 0.33126411f * rgb.y + 0.50000000f * rgb.z + 128.0f);
}

template <typename T>
__inline__ __device__ T rgb_to_cr(vec<3, T> rgb) {
  return ConvertSat<T>(0.50000000f * rgb.x - 0.41868759f * rgb.y - 0.08131241f * rgb.z + 128.0f);
}

template <typename T>
__inline__ __device__ vec<2, T> rgb_to_cb_cr(vec<3, T> rgb) {
  return {rgb_to_cb<T>(rgb), rgb_to_cr<T>(rgb)};
}

template <int N, typename T>
__inline__ __device__ vec<N, T> avg4(vec<N, T> a, vec<N, T> b, vec<N, T> c, vec<N, T> d) {
  IMPL_VEC_ELEMENTWISE(ConvertSat<T>((a[i] + b[i] + c[i] + d[i]) * 0.25f));
}

template <int N, typename T>
__inline__ __device__ vec<N, T> avg2(vec<N, T> a, vec<N, T> b) {
  IMPL_VEC_ELEMENTWISE(ConvertSat<T>((a[i] + b[i]) * 0.5f));
}

template <typename T>
__inline__ __device__ vec<3, T> ycbcr_to_rgb(const vec<3, T> ycbcr) {
    float y  = static_cast<float>(ycbcr.x);
    float cb = static_cast<float>(ycbcr.y) - 128.0f;
    float cr = static_cast<float>(ycbcr.z) - 128.0f;
    vec<3, T> rgb;
    rgb.x = ConvertSat<T>(y + 1.402f * cr);
    rgb.y = ConvertSat<T>(y - 0.34413629f * cb - 0.71413629f * cr);
    rgb.z = ConvertSat<T>(y + 1.772f * cb);
    return rgb;
}


template <typename T, bool horz_subsample, bool vert_subsample>
struct YCbCrSubsampled {
  static constexpr int kLumaLen = (1+horz_subsample)*(1+vert_subsample);
  T luma[kLumaLen];
  T cb, cr;
};

template <bool horz_subsample, bool vert_subsample, typename T>
__inline__ __device__
YCbCrSubsampled<T, horz_subsample, vert_subsample>
rgb_to_ycbcr_subsampled(ivec2 offset, const Surface2D<const uint8_t>& in) {
  const auto sampler = make_sampler<DALI_INTERP_NN>(in);
  YCbCrSubsampled<T, horz_subsample, vert_subsample> out;
  int y = offset.y;
  int x = offset.x;
  vec<3, T> rgb[4];
  sampler(rgb[0].v, ivec2(x, y), BorderClamp());
  out.luma[0] = rgb_to_y<T>(rgb[0]);
  vec<3, T> avg_rgb(rgb[0]);
  if (horz_subsample && vert_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
    sampler(rgb[3].v, ivec2(x + 1, y + 1), BorderClamp());
    out.luma[1] = rgb_to_y<T>(rgb[1]);
    out.luma[2] = rgb_to_y<T>(rgb[2]);
    out.luma[3] = rgb_to_y<T>(rgb[3]);
    avg_rgb = avg4(rgb[0], rgb[1], rgb[2], rgb[3]);
  } else if (horz_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    out.luma[1] = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  } else if (vert_subsample) {
    sampler(rgb[1].v, ivec2(x, y + 1), BorderClamp());
    out.luma[1] = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  }

  vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
  out.cb = cbcr.x;
  out.cr = cbcr.y;
  return out;
}

template <int N, typename T>
__inline__ __device__
void write_vec(T* ptr, vec<N, T> v) {
  #pragma unroll
  for (int i = 0; i < N; i++)
    ptr[i] = v[i];
}

template <bool horz_subsample, bool vert_subsample, typename T>
__inline__ __device__
void ycbcr_to_rgb_subsampled(ivec2 offset, const Surface2D<uint8_t>& out,
                             YCbCrSubsampled<T, horz_subsample, vert_subsample> ycbcr) {
  int y = offset.y;
  int x = offset.x;
  write_vec(&out(x, y), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[0], ycbcr.cb, ycbcr.cr)));
  if (horz_subsample && vert_subsample) {
    write_vec(&out(x + 1, y), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[1], ycbcr.cb, ycbcr.cr)));
    write_vec(&out(x, y + 1), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[2], ycbcr.cb, ycbcr.cr)));
    write_vec(&out(x + 1, y + 1), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[3], ycbcr.cb, ycbcr.cr)));
  } else if (horz_subsample) {
    write_vec(&out(x + 1, y), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[1], ycbcr.cb, ycbcr.cr)));
  } else if (vert_subsample) {
    write_vec(&out(x, y + 1), ycbcr_to_rgb(vec<3, T>(ycbcr.luma[1], ycbcr.cb, ycbcr.cr)));
  }
}

template <bool horz_subsample, bool vert_subsample>
__global__ void ChromaSubsampleDistortion(const SampleDesc *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  using T = uint8_t;
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  int y_start = threadIdx.y + block.start.y;
  int x_start = threadIdx.x + block.start.x;
  if (y_start >= block.end.y || x_start >= block.end.x) {
    return;
  }

  const Surface2D<const uint8_t> in = {
    sample.in, sample.size, 3, sample.strides, 1
  };

  const Surface2D<uint8_t> out = {
    sample.out, sample.size, 3, sample.strides, 1
  };

  for (int pos_y = y_start; pos_y < block.end.y; pos_y += blockDim.y) {
    for (int pos_x = x_start; pos_x < block.end.x; pos_x += blockDim.x) {
      int y = pos_y << vert_subsample;
      int x = pos_x << horz_subsample;
      auto ycbcr = rgb_to_ycbcr_subsampled<horz_subsample, vert_subsample, T>(ivec2{x, y}, in);
      ycbcr_to_rgb_subsampled<horz_subsample, vert_subsample, T>(ivec2{x, y}, out, ycbcr);
    }
  }
}

__device__ __inline__ float quantize(float value, float Q_coeff) {
  // Here we are shifting the negative numbers to the positive range,
  // so that the rounded quotient is the closest integer, also for negative numbers.
  // For uint8_t pixels (range 0..255), all DCT coefficients should be
  // below 2048.
  float rounded_quotient = static_cast<int>((value / Q_coeff) + 2048.5f) - 2048.0f;
  return Q_coeff * rounded_quotient;
}

/**
 * @brief Produces JPEG compression artifacts by running the lossy part of
 * JPEG compression and decompression.
 * @param samples Sample descriptors
 * @param blocks Logical block descriptors
 * @param quality_factor Number between 1 (lowest quality) and 100 (highest quality)
 * used to determine the scaling applied to the quantization matrices.
 * Note: quality_factor==100 does not mean lossless compression.
 */
template <bool horz_subsample, bool vert_subsample, bool quantization = true>
__global__ void JpegCompressionDistortion(const SampleDesc *samples,
                                          const kernels::BlockDesc<2> *blocks,
                                          int quality_factor) {
  using T = uint8_t;
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  static constexpr int align_y = 8 << vert_subsample;
  static constexpr int align_x = 8 << horz_subsample;
  int padding_y = align_up(sample.size.y, align_y) - sample.size.y;
  int padding_x = align_up(sample.size.x, align_x) - sample.size.x;
  int aligned_end_y = block.end.y < sample.size.y ? block.end.y : block.end.y + padding_y;
  int aligned_end_x = block.end.x < sample.size.x ? block.end.x : block.end.x + padding_x;

  int y_start = threadIdx.y + block.start.y;
  int x_start = threadIdx.x + block.start.x;
  if (y_start >= aligned_end_y || x_start >= aligned_end_x) {
    return;
  }

  // Assuming CUDA block is 32x16, leading to a 2x4 grid of chroma blocks
  // and up to 4x8 blocks of luma.
  __shared__ float luma_blk[2 << vert_subsample][4 << horz_subsample][8][9];
  __shared__ float cb_blk[2][4][8][9];  // 8+1 to reduce bank conflicts
  __shared__ float cr_blk[2][4][8][9];  // 8+1 to reduce bank conflicts

  int chroma_x = threadIdx.x & 7;  // % 8
  int chroma_y = threadIdx.y & 7;  // % 8
  ivec2 chroma_blk_idx{threadIdx.x >> 3, threadIdx.y >> 3};  // / 8

  int luma_x = threadIdx.x << horz_subsample;
  int luma_y = threadIdx.y << vert_subsample;
  ivec2 luma_blk_idx{luma_x >> 3, luma_y >> 3};  // / 8
  luma_x = luma_x & 7;  // % 8
  luma_y = luma_y & 7;  // % 8

  const Surface2D<const uint8_t> in = {
    sample.in, sample.size, 3, sample.strides, 1
  };

  const Surface2D<uint8_t> out = {
    sample.out, sample.size, 3, sample.strides, 1
  };

  float (&luma)[8][9] = luma_blk[luma_blk_idx.y][luma_blk_idx.x];
  float (&cb)[8][9] = cb_blk[chroma_blk_idx.y][chroma_blk_idx.x];
  float (&cr)[8][9] = cr_blk[chroma_blk_idx.y][chroma_blk_idx.x];

  float q_scale = 1.0f;
  quality_factor = clamp<float>(quality_factor, 1.0f, 99.0f);
  if (1 <= quality_factor && quality_factor < 50) {
    q_scale = 50.0f / quality_factor;
  } else if (50 <= quality_factor && quality_factor < 100) {
    q_scale = 2.0f - (2 * quality_factor / 100.0f);
  }

  for (int pos_y = y_start; pos_y < aligned_end_y; pos_y += blockDim.y) {
    for (int pos_x = x_start; pos_x < aligned_end_x; pos_x += blockDim.x) {
      int y = pos_y << vert_subsample;
      int x = pos_x << horz_subsample;
      ivec2 offset{x, y};

      auto ycbcr = rgb_to_ycbcr_subsampled<horz_subsample, vert_subsample, T>(offset, in);
      // Shifting to [-128, 128] before the DCT.
      cb[chroma_y][chroma_x] = ycbcr.cb - 128.0f;
      cr[chroma_y][chroma_x] = ycbcr.cr - 128.0f;
      for (int i = 0, k = 0; i < vert_subsample+1; i++) {
        for (int j = 0; j < horz_subsample+1; j++, k++) {
          luma[luma_y + i][luma_x + j] = ycbcr.luma[k] - 128.0f;
        }
      }

      __syncthreads();

      if (blockIdx.x == 0 && threadIdx.x==0 && threadIdx.y==0) {
        printf("Luma before DCT:\n");
        for (int i = 0; i < 8; i++) {
          printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n",
                 (int) luma[i][0]+ 128, (int) luma[i][1]+ 128, (int) luma[i][2]+ 128, (int) luma[i][3]+ 128,
                 (int) luma[i][4]+ 128, (int) luma[i][5]+ 128, (int) luma[i][6]+ 128, (int) luma[i][7]+ 128);
        }
        printf("\n");
      }
      __syncthreads();

      static constexpr int col_stride = 1;
      static constexpr int row_stride = 9;

      if (chroma_x <= vert_subsample) {  // 1 or 2 rows depending on vertical subsampling
        dct_fwd_8x8_1d<col_stride>(&luma[luma_y + chroma_x][0]);
      }

      if (chroma_x == 0) {  // once per row
        dct_fwd_8x8_1d<col_stride>(&cb[chroma_y][0]);
        dct_fwd_8x8_1d<col_stride>(&cr[chroma_y][0]);
      }

      __syncthreads();

      if (chroma_y <= horz_subsample) {  // 1 or 2 columns depending on horizontal subsampling
        dct_fwd_8x8_1d<row_stride>(&luma[0][luma_x + chroma_y]);
      }
      if (chroma_y == 0) {  // once per column
        dct_fwd_8x8_1d<row_stride>(&cb[0][chroma_x]);
        dct_fwd_8x8_1d<row_stride>(&cr[0][chroma_x]);
      }

      __syncthreads();

      if (blockIdx.x == 0 && threadIdx.x==0 && threadIdx.y==0) {
        printf("Luma before quantization:\n");
        for (int i = 0; i < 8; i++) {
          printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n",
                 (int) luma[i][0], (int) luma[i][1], (int) luma[i][2], (int) luma[i][3],
                 (int) luma[i][4], (int) luma[i][5], (int) luma[i][6], (int) luma[i][7]);
        }
        printf("\n");
      }
      __syncthreads();

      if (quantization) {
        float q_chroma_coeff = ConvertSat<int>(q_scale * Q_chroma[chroma_y][chroma_x]);
        if (q_chroma_coeff < 1)
          q_chroma_coeff = 1;
        cb[chroma_y][chroma_x] = quantize(cb[chroma_y][chroma_x], q_chroma_coeff);
        cr[chroma_y][chroma_x] = quantize(cr[chroma_y][chroma_x], q_chroma_coeff);
        for (int i = 0, k = 0; i < vert_subsample+1; i++) {
          for (int j = 0; j < horz_subsample+1; j++, k++) {
            float q_coeff_luma = ConvertSat<int>(q_scale * Q_luma[luma_y + i][luma_x + j]);
            if (q_coeff_luma < 1)
              q_coeff_luma = 1;
            luma[luma_y + i][luma_x + j] = quantize(luma[luma_y + i][luma_x + j], q_coeff_luma);
          }
        }
      }
      __syncthreads();

      if (blockIdx.x == 0 && threadIdx.x==0 && threadIdx.y==0) {
        printf("Luma after quantization:\n");
        for (int i = 0; i < 8; i++) {
          printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n",
                 (int) luma[i][0], (int) luma[i][1], (int) luma[i][2], (int) luma[i][3],
                 (int) luma[i][4], (int) luma[i][5], (int) luma[i][6], (int) luma[i][7]);
        }
        printf("\n");
      }
      __syncthreads();

      if (chroma_y <= horz_subsample) {  // 1 or 2 columns depending on horizontal subsample
        dct_inv_8x8_1d<row_stride>(&luma[0][luma_x + chroma_y]);
      }
      if (chroma_y == 0) {  // once per column
        dct_inv_8x8_1d<row_stride>(&cb[0][chroma_x]);
        dct_inv_8x8_1d<row_stride>(&cr[0][chroma_x]);
      }

      __syncthreads();

      if (chroma_x <= vert_subsample) {  // 1 or 2 rows depending on vertical subsampling
        dct_inv_8x8_1d<col_stride>(&luma[luma_y + chroma_x][0]);
      }

      if (chroma_x == 0) {  // once per row
        dct_inv_8x8_1d<col_stride>(&cb[chroma_y][0]);
        dct_inv_8x8_1d<col_stride>(&cr[chroma_y][0]);
      }

      __syncthreads();

      if (blockIdx.x == 0 && threadIdx.x==0 && threadIdx.y==0) {
        printf("Luma after IDCT:\n");
        for (int i = 0; i < 8; i++) {
          printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n",
                 (int) luma[i][0]+ 128, (int) luma[i][1]+ 128, (int) luma[i][2]+ 128, (int) luma[i][3]+ 128,
                 (int) luma[i][4]+ 128, (int) luma[i][5]+ 128, (int) luma[i][6]+ 128, (int) luma[i][7]+ 128);
        }
        printf("\n");
      }
      __syncthreads();

      // If we are in the out-of-bounds region, skip
      if (offset.x >= sample.size.x || offset.y >= sample.size.y) {
        continue;
      }

      YCbCrSubsampled<T, horz_subsample, vert_subsample> out_ycbcr;
       // Shifting to [0, 255] after the inverse DCT.
      out_ycbcr.cb = ConvertSat<T>(cb[chroma_y][chroma_x] + 128.0f);
      out_ycbcr.cr = ConvertSat<T>(cr[chroma_y][chroma_x] + 128.0f);
      for (int i = 0, k = 0; i < vert_subsample+1; i++) {
        for (int j = 0; j < horz_subsample+1; j++, k++) {
          out_ycbcr.luma[k] = ConvertSat<T>(luma[luma_y + i][luma_x + j] + 128.0f);
        }
      }
      ycbcr_to_rgb_subsampled<horz_subsample, vert_subsample, T>(offset, out, out_ycbcr);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_
