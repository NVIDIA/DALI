// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "color_space.h"

#include <cuda_runtime.h>

typedef struct {
    uint8_t r, g, b;
} Rgb;

__constant__ float mat_yuv_to_rgb[3][3] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

__device__ static uint8_t clamp(float x, float lower, float upper) {
    return fminf(fmaxf(x, lower), upper);
}

__device__ inline Rgb pixel_yuv_to_rgb(uint8_t y, uint8_t u, uint8_t v) {
    const int low = 1 << (sizeof(uint8_t) * 8 - 4);
    const int mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low;
    float fu = (int)u - mid;
    float fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;

    return Rgb { 
        clamp(mat_yuv_to_rgb[0][0] * fy + mat_yuv_to_rgb[0][1] * fu + mat_yuv_to_rgb[0][2] * fv, 0.0f, maxf),
        clamp(mat_yuv_to_rgb[1][0] * fy + mat_yuv_to_rgb[1][1] * fu + mat_yuv_to_rgb[1][2] * fv, 0.0f, maxf),
        clamp(mat_yuv_to_rgb[2][0] * fy + mat_yuv_to_rgb[2][1] * fu + mat_yuv_to_rgb[2][2] * fv, 0.0f, maxf)};
}

__global__ static void yuv_to_rgb_kernel(
    uint8_t *yuv, int yuv_pitch, uint8_t *rgb, int rgb_pitch, int width, int height) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height) {
        return;
    }

    uint8_t *src = yuv + x * sizeof(uint8_t) + y * yuv_pitch;

    uint8_t *dst_1 = rgb + x * sizeof(Rgb) + y * rgb_pitch;
    uint8_t *dst_2 = rgb + x * sizeof(Rgb) + (y+1) * rgb_pitch;

    uint8_t luma_1 = *src;
    uint8_t luma_2 = *(src + yuv_pitch);
    uint8_t *chroma = (src + (height - y / 2) * yuv_pitch);

    Rgb pixel_1 = pixel_yuv_to_rgb(luma_1, chroma[0], chroma[1]);
    Rgb pixel_2 = pixel_yuv_to_rgb(luma_2, chroma[0], chroma[1]);

    dst_1[0] = pixel_1.r;
    dst_1[1] = pixel_1.g;
    dst_1[2] = pixel_1.b;

    dst_1[3] = pixel_1.r;
    dst_1[4] = pixel_1.g;
    dst_1[5] = pixel_1.b;

    dst_2[0] = pixel_2.r;
    dst_2[1] = pixel_2.g;
    dst_2[2] = pixel_2.b;

    dst_2[3] = pixel_2.r;
    dst_2[4] = pixel_2.g;
    dst_2[5] = pixel_2.b;

}

void yuv_to_rgb(uint8_t *yuv, int yuv_pitch, uint8_t *rgb, int rgb_pitch, int width, int height, cudaStream_t stream) {
    auto grid_layout = dim3((width + 63) / 32 / 2, (height + 3)); 
    auto block_layout = dim3(32, 2);

    yuv_to_rgb_kernel
        <<<grid_layout, block_layout, 0, stream>>>
        (yuv, yuv_pitch, rgb, rgb_pitch, width, height);
}
