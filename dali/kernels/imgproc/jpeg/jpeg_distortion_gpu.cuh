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
#include <cuda_fp16.h>

#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

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



template <bool horz_subsample, bool vert_subsample>
__global__ void JpegCompressionDistortion(const SampleDesc *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  using T = uint8_t;
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  int y_start = threadIdx.y + block.start.y;
  int x_start = threadIdx.x + block.start.x;
  if (y_start >= block.end.y || x_start >= block.end.x) {
    return;
  }

  // Assuming CUDA block is 32x16, leading to a 2x4 grid of chroma blocks
  // and up to 4x8 blocks of luma.
  __shared__ T luma_blk[2 << vert_subsample][4 << horz_subsample][8][9];
  __shared__ T cb_blk[2][4][8][9];  // 8+1 to reduce bank conflicts
  __shared__ T cr_blk[2][4][8][9];  // 8+1 to reduce bank conflicts

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

  T (&luma)[8][9] = luma_blk[luma_blk_idx.y][luma_blk_idx.x];
  T (&cb)[8][9] = cb_blk[chroma_blk_idx.y][chroma_blk_idx.x];
  T (&cr)[8][9] = cr_blk[chroma_blk_idx.y][chroma_blk_idx.x];

  for (int pos_y = y_start; pos_y < block.end.y; pos_y += blockDim.y) {
    for (int pos_x = x_start; pos_x < block.end.x; pos_x += blockDim.x) {
      int y = pos_y << vert_subsample;
      int x = pos_x << horz_subsample;
      ivec2 offset{x, y};

      auto ycbcr = rgb_to_ycbcr_subsampled<horz_subsample, vert_subsample, T>(offset, in);
      luma[luma_y][luma_x] = ycbcr.luma[0];
      cb[chroma_y][chroma_x] = ycbcr.cb;
      cr[chroma_y][chroma_x] = ycbcr.cr;
      if (horz_subsample && vert_subsample) {
        luma[luma_y][luma_x + 1] = ycbcr.luma[1];
        luma[luma_y + 1][luma_x] = ycbcr.luma[2];
        luma[luma_y + 1][luma_x + 1] = ycbcr.luma[3];
      } else if (horz_subsample) {
        luma[luma_y][luma_x + 1] = ycbcr.luma[1];
      } else if (vert_subsample) {
        luma[luma_y + 1][luma_x] = ycbcr.luma[1];
      }

      __syncthreads();

      // TODO(janton): DCT + quantization + inv DCT

      YCbCrSubsampled<T, horz_subsample, vert_subsample> out_ycbcr;
      out_ycbcr.luma[0] = luma[luma_y][luma_x];
      out_ycbcr.cb = cb[chroma_y][chroma_x];
      out_ycbcr.cr = cr[chroma_y][chroma_x];
      if (horz_subsample && vert_subsample) {
        out_ycbcr.luma[1] = luma[luma_y][luma_x + 1];
        out_ycbcr.luma[2] = luma[luma_y + 1][luma_x];
        out_ycbcr.luma[3] = luma[luma_y + 1][luma_x + 1];
      } else if (horz_subsample) {
        out_ycbcr.luma[1] = luma[luma_y][luma_x + 1];
      } else if (vert_subsample) {
        out_ycbcr.luma[1] = luma[luma_y + 1][luma_x];
      }
      ycbcr_to_rgb_subsampled<horz_subsample, vert_subsample, T>(offset, out, out_ycbcr);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_
