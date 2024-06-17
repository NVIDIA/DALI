// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "decoder/color_space.h"

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

namespace dali {

template <bool full_range>
__global__ static void yuv_to_rgb_kernel(
    const uint8_t *yuv, int yuv_pitch, uint8_t *rgb, int rgb_pitch, int width, int height) {
    int halfx = (threadIdx.x + blockIdx.x * blockDim.x);
    int halfy = (threadIdx.y + blockIdx.y * blockDim.y);
    int x = 2 * halfx;
    int y = 2 * halfy;
    if (x >= width || y >= height) {
        return;
    }

    kernels::Surface2D<const uint8_t> Y_surf, UV_surf;
    kernels::Surface2D<uint8_t> RGB;
    const uint8_t *chroma = yuv + height * yuv_pitch;

    Y_surf  = { yuv,    width,     height,     1, 1, yuv_pitch, 1 };
    UV_surf = { chroma, width / 2, height / 2, 2, 2, yuv_pitch, 1 };

    RGB = { rgb, width, height, 3, 3, rgb_pitch, 1 };

    auto Y = kernels::make_sampler<DALI_INTERP_NN>(Y_surf);
    auto UV = kernels::make_sampler<DALI_INTERP_LINEAR>(UV_surf);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        float cy = halfy + i * 0.5f + 0.25f;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            float cx = halfx + j * 0.5f + 0.25f;
            u8vec3 yuv_val;
            yuv_val[0] = Y.at(ivec2{x + j, y + i}, 0, kernels::BorderClamp());

            UV(&yuv_val[1], vec2(cx, cy), kernels::BorderClamp());

            u8vec3 rgb_val;
            if (full_range)
                rgb_val = dali::kernels::color::jpeg::ycbcr_to_rgb<uint8_t>(yuv_val);
            else
                rgb_val = dali::kernels::color::itu_r_bt_601::ycbcr_to_rgb<uint8_t>(yuv_val);

            RGB({x + j, y + i, 0}) = rgb_val.x;
            RGB({x + j, y + i, 1}) = rgb_val.y;
            RGB({x + j, y + i, 2}) = rgb_val.z;
        }
    }
}

}  // namespace dali

void yuv_to_rgb(uint8_t *yuv, int yuv_pitch, uint8_t *rgb, int rgb_pitch, int width, int height,
                bool full_range, cudaStream_t stream) {
    auto grid_layout = dim3((width + 63) / 32 / 2, (height + 3));
    auto block_layout = dim3(32, 2);

    if (full_range) {
        dali::yuv_to_rgb_kernel<true>
            <<<grid_layout, block_layout, 0, stream>>>
            (yuv, yuv_pitch, rgb, rgb_pitch, width, height);
    } else {
        dali::yuv_to_rgb_kernel<false>
            <<<grid_layout, block_layout, 0, stream>>>
            (yuv, yuv_pitch, rgb, rgb_pitch, width, height);
    }
}
