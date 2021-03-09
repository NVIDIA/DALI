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
  vec<2, int> in_size, out_y_size, out_chroma_size;
  vec<2, int64_t> in_strides, out_y_strides, out_chroma_strides;
};

template <typename T>
__inline__ __device__ T rgb_to_y(const vec<3, uint8_t> rgb) {
  return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
}

template <typename T>
__inline__ __device__ vec<3, T> rgb_to_ycbcr(const vec<3, uint8_t> rgb) {
  float y  =  rgb_to_y<T>(rgb);
  float cb = -0.16873589f * rgb.x - 0.33126411f * rgb.y + 0.50000000f * rgb.z + 128.0f;
  float cr =  0.50000000f * rgb.x - 0.41868759f * rgb.y - 0.08131241f * rgb.z + 128.0f;
  cr = clamp<float>(cr, 0, 255);
  return vec<3, T>{y, cb, cr};
}

template <bool horz_subsample, bool vert_subsample, typename T = uint8_t, int in_nchannels = 3>
__global__ void RGBToYCbCrChromaSubsample(const SampleDesc<T> *samples,
                                          const kernels::BlockDesc<2> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  const Surface2D<const uint8_t> in = {
    sample.in, sample.in_size.x, sample.in_size.y, in_nchannels,
    sample.in_strides.x, sample.in_strides.y, 1
  };

  const auto sampler = make_sampler<DALI_INTERP_LINEAR>(in);

  const Surface2D<T> out_y = {
    sample.out_y, sample.out_y_size.x, sample.out_y_size.y, 1,
    sample.out_y_strides.x, sample.out_y_strides.y, 1
  };

  const Surface2D<T> out_cb = {
    sample.out_cb, sample.out_chroma_size.x, sample.out_chroma_size.y, 1,
    sample.out_chroma_strides.x, sample.out_chroma_strides.y, 1
  };

  const Surface2D<T> out_cr = {
    sample.out_cr, sample.out_chroma_size.x, sample.out_chroma_size.y, 1,
    sample.out_chroma_strides.x, sample.out_chroma_strides.y, 1
  };

  for (int y = threadIdx.y + block.start.y; y < block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
      int64_t in_x = x << horz_subsample;
      int64_t in_y = y << vert_subsample;

      u8vec3 rgb[4];
      sampler(rgb[0].v, ivec2(in_x, in_y), BorderClamp());
      out_y(x, y) = rgb_to_y<T>(rgb[0]);
      u8vec3 avg_rgb = rgb[0];
      if (horz_subsample && vert_subsample) {
        sampler(rgb[1].v, ivec2(in_x + 1, in_y), BorderClamp());
        sampler(rgb[2].v, ivec2(in_x, in_y + 1), BorderClamp());
        sampler(rgb[3].v, ivec2(in_x + 1, in_y + 1), BorderClamp());
        out_y(x + 1, y) = rgb_to_y<T>(rgb[1]);
        out_y(x, y + 1) = rgb_to_y<T>(rgb[2]);
        out_y(x + 1, y + 1) = rgb_to_y<T>(rgb[3]);
        sampler(avg_rgb.v, vec2(in_x + 1.0f, in_y + 1.0f), BorderClamp());  // average
      } else if (horz_subsample) {
        sampler(rgb[1].v, ivec2(in_x + 1, in_y), BorderClamp());
        out_y(x + 1, y) = rgb_to_y<T>(rgb[1]);
        sampler(avg_rgb.v, vec2(in_x + 1.0f, in_y), BorderClamp());  // average
      } else if (vert_subsample) {
        sampler(rgb[2].v, ivec2(in_x, in_y + 1), BorderClamp());
        out_y(x, y + 1) = rgb_to_y<T>(rgb[2]);
        sampler(avg_rgb.v, vec2(in_x, in_y + 1.0f), BorderClamp());  // average
      }

      u8vec3 ycbcr = rgb_to_ycbcr<T>(avg_rgb);
      out_cb(x, y) = ycbcr.y;
      out_cr(x, y) = ycbcr.z;
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_JPEG_CHROMA_SUBSAMPLE_GPU_H_
