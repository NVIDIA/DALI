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

#ifndef DALI_KERNELS_IMGPROC_JPEG_CHROMA_SUBSAMPLE_GPU_H_
#define DALI_KERNELS_IMGPROC_JPEG_CHROMA_SUBSAMPLE_GPU_H_

#include <cuda_runtime_api.h>

#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

template <typename T = uint8_t>
struct SampleDesc {
  const uint8_t *in;  // rgb
  T *out_y, *out_cb, *out_cr;
  ivec<2> in_size, out_y_size, out_chroma_size;
  i64vec<2> in_strides, out_y_strides, out_chroma_strides;
};

template <typename T>
__inline__ __device__ T rgb_to_y(const vec<3, uint8_t> rgb) {
  return ConvertSat<T>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
}

template <typename T>
__inline__ __device__ vec<3, T> rgb_to_ycbcr(const vec<3, uint8_t> rgb) {
  vec<3, T> ycbcr;
  ycbcr.x = rgb_to_y<T>(rgb);
  ycbcr.y = ConvertSat<T>(-0.16873589f * rgb.x - 0.33126411f * rgb.y + 0.50000000f * rgb.z + 128.0f);
  ycbcr.z = ConvertSat<T>(0.50000000f * rgb.x - 0.41868759f * rgb.y - 0.08131241f * rgb.z + 128.0f);
  return ycbcr;
}

template <bool horz_subsample, bool vert_subsample, typename T = uint8_t, int in_nchannels = 3>
__global__ void RGBToYCbCrChromaSubsample(const SampleDesc<T> *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  int y_start = threadIdx.y + block.start.y;
  int x_start = threadIdx.x + block.start.x;
  if (y_start >= block.end.y || x_start >= block.end.x) {
    return;
  }

  const Surface2D<const uint8_t> in = {
    sample.in, sample.in_size.x, sample.in_size.y, in_nchannels,
    sample.in_strides.x, sample.in_strides.y, 1
  };

  const auto sampler = make_sampler<DALI_INTERP_LINEAR>(in);

  const Surface2D<T> out_y = {
    sample.out_y, sample.out_y_size, 1, sample.out_y_strides, 1
  };

  const Surface2D<T> out_cb = {
    sample.out_cb, sample.out_chroma_size, 1, sample.out_chroma_strides, 1
  };

  const Surface2D<T> out_cr = {
    sample.out_cr, sample.out_chroma_size, 1, sample.out_chroma_strides, 1
  };

  for (int chroma_y = y_start; chroma_y < block.end.y; chroma_y += blockDim.y) {
    for (int chroma_x = x_start; chroma_x < block.end.x; chroma_x += blockDim.x) {
      int x = chroma_x << horz_subsample;
      int y = chroma_y << vert_subsample;

      u8vec3 rgb[4];
      sampler(rgb[0].v, ivec2(x, y), BorderClamp());
      out_y(x, y) = rgb_to_y<T>(rgb[0]);
      u8vec3 avg_rgb = rgb[0];
      if (horz_subsample && vert_subsample) {
        sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
        sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
        sampler(rgb[3].v, ivec2(x + 1, y + 1), BorderClamp());
        out_y(x + 1, y) = rgb_to_y<T>(rgb[1]);
        out_y(x, y + 1) = rgb_to_y<T>(rgb[2]);
        out_y(x + 1, y + 1) = rgb_to_y<T>(rgb[3]);
        sampler(avg_rgb.v, vec2(x + 1.0f, y + 1.0f), BorderClamp());  // average
      } else if (horz_subsample) {
        sampler(rgb[1].v, ivec2(x + 1, y), BorderClamp());
        out_y(x + 1, y) = rgb_to_y<T>(rgb[1]);
        sampler(avg_rgb.v, vec2(x + 1.0f, y + 0.5f), BorderClamp());  // average
      } else if (vert_subsample) {
        sampler(rgb[2].v, ivec2(x, y + 1), BorderClamp());
        out_y(x, y + 1) = rgb_to_y<T>(rgb[2]);
        sampler(avg_rgb.v, vec2(x + 0.5f, y + 1.0f), BorderClamp());  // average
      }

      auto ycbcr = rgb_to_ycbcr<T>(avg_rgb);
      out_cb(chroma_x, chroma_y) = ycbcr.y;
      out_cr(chroma_x, chroma_y) = ycbcr.z;
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_CHROMA_SUBSAMPLE_GPU_H_
