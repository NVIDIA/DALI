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
__inline__ __device__ vec<N> avg4(vec<N, T> a, vec<N, T> b, vec<N, T> c, vec<N, T> d) {
  IMPL_VEC_ELEMENTWISE((a[i] + b[i] + c[i] + d[i]) * 0.25f);
}

template <int N, typename T>
__inline__ __device__ vec<N> avg2(vec<N, T> a, vec<N, T> b) {
  IMPL_VEC_ELEMENTWISE((a[i] + b[i]) * 0.5f);
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
void rgb_to_ycbcr_chroma_subsample(ivec2 blk_offset, ivec2 offset,
                                   const Surface2D<T>& luma,
                                   const Surface2D<T>& cb,
                                   const Surface2D<T>& cr,
                                   const Surface2D<const uint8_t>& in) {
  const auto sampler = make_sampler<DALI_INTERP_NN>(in);
  int chroma_y = blk_offset.y;
  int chroma_x = blk_offset.x;
  int luma_y = chroma_y << vert_subsample;
  int luma_x = chroma_x << horz_subsample;
  int y = offset.y;
  int x = offset.x;

  vec<3, T> rgb[4];
  sampler(rgb[0].v, ivec2(x, y), BorderClamp());
  luma(luma_x, luma_y) = rgb_to_y<T>(rgb[0]);

  vec<3, T> avg_rgb(rgb[0]);
  if (horz_subsample && vert_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
    sampler(rgb[3].v, ivec2(x + 1, y + 1), BorderClamp());
    luma(luma_x + 1, luma_y) = rgb_to_y<T>(rgb[1]);
    luma(luma_x, luma_y + 1) = rgb_to_y<T>(rgb[2]);
    luma(luma_x + 1, luma_y + 1) = rgb_to_y<T>(rgb[3]);
    avg_rgb = avg4(rgb[0], rgb[1], rgb[2], rgb[3]);
  } else if (horz_subsample) {
    sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
    luma(luma_x + 1, luma_y) = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  } else if (vert_subsample) {
    sampler(rgb[1].v, ivec2(x, y + 1), BorderClamp());
    luma(luma_x, luma_y + 1) = rgb_to_y<T>(rgb[1]);
    avg_rgb = avg2(rgb[0], rgb[1]);
  }

  vec<2, T> cbcr = rgb_to_cb_cr<T>(avg_rgb);
  cb(chroma_x, chroma_y) = cbcr.x;
  cr(chroma_x, chroma_y) = cbcr.y;
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
void ycbcr_to_rgb_chroma_subsample(ivec2 blk_offset, ivec2 offset,
                                   const Surface2D<uint8_t>& out,
                                   const Surface2D<T>& luma,
                                   const Surface2D<T>& cb,
                                   const Surface2D<T>& cr) {
  int chroma_y = blk_offset.y;
  int chroma_x = blk_offset.x;
  int luma_y = chroma_y << vert_subsample;
  int luma_x = chroma_x << horz_subsample;
  int y = offset.y;
  int x = offset.x;
  T cb_val = cb(chroma_x, chroma_y);
  T cr_val = cr(chroma_x, chroma_y);

  write_vec(&out(x, y),
            ycbcr_to_rgb(vec<3, T>(luma(luma_x, luma_y), cb_val, cr_val)));
  if (horz_subsample && vert_subsample) {
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(luma(luma_x + 1, luma_y), cb_val, cr_val)));
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_x, luma_y + 1), cb_val, cr_val)));
    write_vec(&out(x + 1, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_x + 1, luma_y + 1), cb_val, cr_val)));
  } else if (horz_subsample) {
    write_vec(&out(x + 1, y),
              ycbcr_to_rgb(vec<3, T>(luma(luma_x + 1, luma_y), cb_val, cr_val)));
  } else if (vert_subsample) {
    write_vec(&out(x, y + 1),
              ycbcr_to_rgb(vec<3, T>(luma(luma_x, luma_y + 1), cb_val, cr_val)));
  }
}


template <bool horz_subsample, bool vert_subsample>
__global__ void ChromaSubsampleDistortion(const SampleDesc *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  using T = uint8_t;
  constexpr int luma_blk_h = 8 << vert_subsample;
  constexpr int luma_blk_w = 8 << horz_subsample;
  constexpr ivec<2> luma_blk_sz{luma_blk_w, luma_blk_h};
  constexpr i64vec<2> luma_blk_strides{1, luma_blk_w};
  constexpr ivec<2> chroma_blk_sz{8, 8};
  constexpr i64vec<2> chroma_blk_strides{1, 8};

  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  int y_start = threadIdx.y + block.start.y;
  int x_start = threadIdx.x + block.start.x;
  if (y_start >= block.end.y || x_start >= block.end.x) {
    return;
  }

  // Assuming CUDA block has:
  // - width 32, leads to 4 horizontal blocks of 8
  // - height 8, so a single block 8x8 fits vertically
  __shared__ T luma_blk[4][luma_blk_h][luma_blk_w];
  __shared__ T cb_blk[4][8][8];
  __shared__ T cr_blk[4][8][8];

  int blk_idx = threadIdx.x / 8;
  int local_x = threadIdx.x % 8;
  int local_y = threadIdx.y;

  const Surface2D<const uint8_t> in = {
    sample.in, sample.size, 3, sample.strides, 1
  };

  const Surface2D<uint8_t> out = {
    sample.out, sample.size, 3, sample.strides, 1
  };

  const Surface2D<T> luma = {
    &luma_blk[blk_idx][0][0], luma_blk_sz, 1, luma_blk_strides, 1
  };

  const Surface2D<T> cb = {
    &cb_blk[blk_idx][0][0], chroma_blk_sz, 1, chroma_blk_strides, 1
  };

  const Surface2D<T> cr = {
    &cr_blk[blk_idx][0][0], chroma_blk_sz, 1, chroma_blk_strides, 1
  };

  for (int chroma_y = y_start; chroma_y < block.end.y; chroma_y += blockDim.y) {
    for (int chroma_x = x_start; chroma_x < block.end.x; chroma_x += blockDim.x) {
      int y = chroma_y << vert_subsample;
      int x = chroma_x << horz_subsample;

      ivec2 blk_offset{local_x, local_y};
      ivec2 offset{x, y};

      rgb_to_ycbcr_chroma_subsample<horz_subsample, vert_subsample>(
          blk_offset, offset, luma, cb, cr, in);

      __syncthreads();

      ycbcr_to_rgb_chroma_subsample<horz_subsample, vert_subsample>(
          blk_offset, offset, out, luma, cb, cr);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_JPEG_ARTIFACTS_GPU_H_
