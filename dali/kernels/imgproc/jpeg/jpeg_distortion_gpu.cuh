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

template <bool horz_subsample, bool vert_subsample, typename T>
__inline__ __device__
void rgb_to_ycbcr_chroma_subsample(ivec2 luma_offset, ivec2 chroma_offset, ivec2 offset,
                                   const Surface2D<T>& luma,
                                   const Surface2D<T>& cb,
                                   const Surface2D<T>& cr,
                                   const Surface2D<const uint8_t>& in) {
  const auto sampler = make_sampler<DALI_INTERP_NN>(in);
  int y = offset.y;
  int x = offset.x;
  vec<3, T> rgb[4];
  sampler(rgb[0].v, ivec2(x, y), BorderClamp());
  luma(luma_offset.x, luma_offset.y) = rgb_to_y<T>(rgb[0]);
  vec<3, T> avg_rgb(rgb[0]);
  if (horz_subsample && vert_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
    sampler(rgb[3].v, ivec2(x + 1, y + 1), BorderClamp());
    luma(luma_offset.x + 1, luma_offset.y) = rgb_to_y<T>(rgb[1]);
    luma(luma_offset.x, luma_offset.y + 1) = rgb_to_y<T>(rgb[2]);
    luma(luma_offset.x + 1, luma_offset.y + 1) = rgb_to_y<T>(rgb[3]);
    avg_rgb = avg4(rgb[0], rgb[1], rgb[2], rgb[3]);
  } else if (horz_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    luma(luma_offset.x + 1, luma_offset.y) = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  } else if (vert_subsample) {
    sampler(rgb[1].v, ivec2(x, y + 1), BorderClamp());
    luma(luma_offset.x, luma_offset.y + 1) = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  }

  vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
  cb(chroma_offset.x, chroma_offset.y) = cbcr.x;
  cr(chroma_offset.x, chroma_offset.y) = cbcr.y;
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
void ycbcr_to_rgb_chroma_subsample(ivec2 luma_offset, ivec2 chroma_offset, ivec2 offset,
                                   const Surface2D<uint8_t>& out,
                                   const Surface2D<T>& luma,
                                   const Surface2D<T>& cb,
                                   const Surface2D<T>& cr) {
  int y = offset.y;
  int x = offset.x;
  T cb_val = cb(chroma_offset.x, chroma_offset.y);
  T cr_val = cr(chroma_offset.x, chroma_offset.y);
  write_vec(&out(x, y),
            ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x, luma_offset.y), cb_val, cr_val)));
  if (horz_subsample && vert_subsample) {
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x + 1, luma_offset.y), cb_val, cr_val)));
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x, luma_offset.y + 1), cb_val, cr_val)));
    write_vec(&out(x + 1, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x + 1, luma_offset.y + 1), cb_val, cr_val)));
  } else if (horz_subsample) {
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x + 1, luma_offset.y), cb_val, cr_val)));
  } else if (vert_subsample) {
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_offset.x, luma_offset.y + 1), cb_val, cr_val)));
  }
}

template <bool horz_subsample, bool vert_subsample>
__inline__ __device__
void rgb_to_ycbcr_to_rgb_chroma_subsample(ivec2 offset,
                                          const Surface2D<uint8_t>& out,
                                          const Surface2D<const uint8_t>& in) {
  using T = uint8_t;
  const auto sampler = make_sampler<DALI_INTERP_NN>(in);
  int y = offset.y;
  int x = offset.x;
  vec<3, T> rgb[4];
  sampler(rgb[0].v, ivec2(x, y), BorderClamp());
  if (horz_subsample && vert_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
    sampler(rgb[3].v, ivec2(x + 1, y + 1), BorderClamp());
    vec<3, T> avg_rgb = avg4(rgb[0], rgb[1], rgb[2], rgb[3]);
    vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
    write_vec(&out(x, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[0]), cbcr.x, cbcr.y)));
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[1]), cbcr.x, cbcr.y)));
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[2]), cbcr.x, cbcr.y)));
    write_vec(&out(x + 1, y + 1),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[3]), cbcr.x, cbcr.y)));
  } else if (horz_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    vec<3, T> avg_rgb = avg2(rgb[0], rgb[1]);
    vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
    write_vec(&out(x, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[0]), cbcr.x, cbcr.y)));
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[1]), cbcr.x, cbcr.y)));
  } else if (vert_subsample) {
    sampler(rgb[1].v, ivec2(x, y + 1), BorderClamp());
    vec<3, T> avg_rgb = avg2(rgb[0], rgb[1]);
    vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
    write_vec(&out(x, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[0]), cbcr.x, cbcr.y)));
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[1]), cbcr.x, cbcr.y)));
  } else {
    vec<2, T> cbcr = rgb_to_cb_cr<T>(rgb[0]);
    write_vec(&out(x, y),
              ycbcr_to_rgb(vec<3, T>(rgb_to_y<T>(rgb[0]), cbcr.x, cbcr.y)));
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

  for (int chroma_y = y_start; chroma_y < block.end.y; chroma_y += blockDim.y) {
    for (int chroma_x = x_start; chroma_x < block.end.x; chroma_x += blockDim.x) {
      int y = chroma_y << vert_subsample;
      int x = chroma_x << horz_subsample;
      rgb_to_ycbcr_to_rgb_chroma_subsample<horz_subsample, vert_subsample>(ivec2{x, y}, out, in);
    }
  }
}



template <bool horz_subsample, bool vert_subsample>
__global__ void JpegCompressionDistortion(const SampleDesc *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  using T = uint8_t;
  constexpr ivec<2> blk_sz{8, 8};
  constexpr i64vec<2> blk_strides{1, 8};

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

  int chroma_x = threadIdx.x;
  int chroma_y = threadIdx.y;
  ivec2 chroma_blk_idx{chroma_x >> 3, chroma_y >> 3};  // / 8
  ivec2 chroma_offset{chroma_x & 7, chroma_y & 7};  // % 8

  int luma_x = threadIdx.x << horz_subsample;
  int luma_y = threadIdx.y << vert_subsample;
  ivec2 luma_blk_idx{luma_x >> 3, luma_y >> 3};  // / 8
  ivec2 luma_offset{luma_x & 7, luma_y & 7};  // % 8

  const Surface2D<const uint8_t> in = {
    sample.in, sample.size, 3, sample.strides, 1
  };

  const Surface2D<uint8_t> out = {
    sample.out, sample.size, 3, sample.strides, 1
  };

  const Surface2D<T> luma = {
    &luma_blk[luma_blk_idx.y][luma_blk_idx.x][0][0], blk_sz, 1, blk_strides, 1
  };

  const Surface2D<T> cb = {
    &cb_blk[chroma_blk_idx.y][chroma_blk_idx.x][0][0], blk_sz, 1, blk_strides, 1
  };

  const Surface2D<T> cr = {
    &cr_blk[chroma_blk_idx.y][chroma_blk_idx.x][0][0], blk_sz, 1, blk_strides, 1
  };

  for (int chroma_y = y_start; chroma_y < block.end.y; chroma_y += blockDim.y) {
    for (int chroma_x = x_start; chroma_x < block.end.x; chroma_x += blockDim.x) {
      int y = chroma_y << vert_subsample;
      int x = chroma_x << horz_subsample;
      ivec2 offset{x, y};

      rgb_to_ycbcr_chroma_subsample<horz_subsample, vert_subsample>(
          luma_offset, chroma_offset, offset, luma, cb, cr, in);

      __syncthreads();

      // TODO(janton): DCT + quantization + inv DCT

      ycbcr_to_rgb_chroma_subsample<horz_subsample, vert_subsample>(
          luma_offset, chroma_offset, offset, out, luma, cb, cr);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_
