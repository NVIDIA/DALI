// Copyright (c) 2022, 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/color_space.h"

#include <cuda_runtime.h>
#include "dali/core/static_switch.h"
#include "dali/core/cuda_error.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

namespace dali {

template <typename Out, VideoColorSpaceConversionType conversion_type, bool normalized_range = false>
__global__ static void VideoColorSpaceConversionKernel(
    Out *out, int out_pitch, const uint8_t *yuv, int yuv_pitch, int height, int width) {
    int halfx = (threadIdx.x + blockIdx.x * blockDim.x);
    int halfy = (threadIdx.y + blockIdx.y * blockDim.y);
    int x = 2 * halfx;
    int y = 2 * halfy;
    if (x >= width || y >= height) {
        return;
    }

    kernels::Surface2D<const uint8_t> Y_surf, UV_surf;
    kernels::Surface2D<Out> output;
    const uint8_t *chroma = yuv + height * yuv_pitch;

    Y_surf  = { yuv,    width,     height,     1, 1, yuv_pitch, 1 };
    UV_surf = { chroma, width / 2, height / 2, 2, 2, yuv_pitch, 1 };

    output = { out, width, height, 3, 3, out_pitch, 1 };

    auto Y = kernels::make_sampler<DALI_INTERP_NN>(Y_surf);
    auto UV = kernels::make_sampler<DALI_INTERP_LINEAR>(UV_surf);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        float cy = halfy + i * 0.5f + 0.25f;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float cx = halfx + j * 0.5f + 0.25f;
            vec3 yuv_val;
            yuv_val[0] = Y.at(ivec2{x + j, y + i}, 0, kernels::BorderClamp());

            UV(&yuv_val[1], vec2(cx, cy), kernels::BorderClamp());

            yuv_val *= 1.0f / 255.0f;

            vec3 out_val;
            switch (conversion_type) {
              case VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE:
                out_val = dali::kernels::color::jpeg::ycbcr_to_rgb<float>(yuv_val);
                break;
              case VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB:
                out_val = dali::kernels::color::itu_r_bt_601::ycbcr_to_rgb<float>(yuv_val);
                break;
              case VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE:
                out_val = yuv_val;
                break;
              default:
                assert(false);
            }
            if (normalized_range) {
              output({x + j, y + i, 0}) = ConvertSatNorm<Out>(out_val.x);
              output({x + j, y + i, 1}) = ConvertSatNorm<Out>(out_val.y);
              output({x + j, y + i, 2}) = ConvertSatNorm<Out>(out_val.z);
            } else {
              out_val *= 255.0f;
              output({x + j, y + i, 0}) = ConvertSat<Out>(out_val.x);
              output({x + j, y + i, 1}) = ConvertSat<Out>(out_val.y);
              output({x + j, y + i, 2}) = ConvertSat<Out>(out_val.z);
            }
        }
    }

}

template <typename Out>
void VideoColorSpaceConversionImpl(Out *out, int out_pitch, const uint8_t *yuv, int yuv_pitch,
                                   int height, int width, VideoColorSpaceConversionType conversion_type,
                                   bool normalized_range, cudaStream_t stream) {
    auto grid_layout = dim3((width + 63) / 32 / 2, (height + 3));
    auto block_layout = dim3(32, 2);

    BOOL_SWITCH(
        normalized_range, static_normalized_range,
        (VALUE_SWITCH(static_cast<int>(conversion_type), static_conversion_type, (
      VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB,
      VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE,
      VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE), (
        VideoColorSpaceConversionKernel<Out, static_conversion_type, static_normalized_range>
        <<<grid_layout, block_layout, 0, stream>>>(out, out_pitch, yuv, yuv_pitch, height, width);
        ), (DALI_FAIL("wrong value")));));
    CUDA_CALL(cudaGetLastError());
}

void VideoColorSpaceConversion(uint8_t *out, int out_pitch, const uint8_t *yuv, int yuv_pitch,
                               int height, int width, VideoColorSpaceConversionType conversion_type,
                               bool normalized_range, cudaStream_t stream) {
  VideoColorSpaceConversionImpl(out, out_pitch, yuv, yuv_pitch, height, width, conversion_type,
                                normalized_range, stream);
}

void VideoColorSpaceConversion(float *out, int out_pitch, const uint8_t *yuv, int yuv_pitch,
                               int height, int width, VideoColorSpaceConversionType conversion_type,
                               bool normalized_range, cudaStream_t stream) {
  VideoColorSpaceConversionImpl(out, out_pitch, yuv, yuv_pitch, height, width, conversion_type,
                                normalized_range, stream);
}

}  // namespace dali
