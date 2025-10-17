// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "dali/core/cuda_error.h"
#include "dali/core/math_util.h"
#include "dali/core/util.h"

#define CV_HEX_CONST(x) __builtin_bit_cast(double, x)

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L100
// 0.412453, 0.357580, 0.180423,
// 0.212671, 0.715160, 0.072169,
// 0.019334, 0.119193, 0.950227
#define CV_RGB_XR CV_HEX_CONST(0x3fda65a14488c60d)  // 0.412453
#define CV_RGB_XG CV_HEX_CONST(0x3fd6e297396d0918)  // 0.357580
#define CV_RGB_XB CV_HEX_CONST(0x3fc71819d2391d58)  // 0.180423

#define CV_RGB_YR CV_HEX_CONST(0x3fcb38cda6e75ff6)  // 0.212673
#define CV_RGB_YG CV_HEX_CONST(0x3fe6e297396d0918)  // 0.715160
#define CV_RGB_YB CV_HEX_CONST(0x3fb279aae6c8f755)  // 0.072169

#define CV_RGB_ZR CV_HEX_CONST(0x3f93cc4ac6cdaf4b)  // 0.019334
#define CV_RGB_ZG CV_HEX_CONST(0x3fbe836eb4e98138)  // 0.119193
#define CV_RGB_ZB CV_HEX_CONST(0x3fee68427418d691)  // 0.950227

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L116
//  3.240479, -1.53715, -0.498535,
// -0.969256, 1.875991, 0.041556,
//  0.055648, -0.204043, 1.057311
#define CV_LAB_XR CV_HEX_CONST(0x4009ec804102ff8f)  // 3.240479
#define CV_LAB_XG CV_HEX_CONST(0xbff8982a9930be0e)  // -1.53715
#define CV_LAB_XB CV_HEX_CONST(0xbfdfe7ff583a53b9)  // -0.498535
#define CV_LAB_YR CV_HEX_CONST(0xbfef042528ae74f3)  // -0.969256
#define CV_LAB_YG CV_HEX_CONST(0x3ffe040f23897204)  // 1.875991
#define CV_LAB_YB CV_HEX_CONST(0x3fa546d3f9e7b80b)  // 0.041556
#define CV_LAB_ZR CV_HEX_CONST(0x3fac7de5082cf52c)  // 0.055648
#define CV_LAB_ZG CV_HEX_CONST(0xbfca1e14bdfd2631)  // -0.204043
#define CV_LAB_ZB CV_HEX_CONST(0x3ff0eabef06b3786)  // 1.057311

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L940
#define D65_WHITE_X CV_HEX_CONST(0x3fee6a22b3892ee8)  // 0.950456
#define D65_WHITE_Y 1.0f                              // 1.000000
#define D65_WHITE_Z CV_HEX_CONST(0x3ff16b8950763a19)  // 1.089058

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1010
#define GAMMA_THRESHOLD (809.0f / 20000.0f)         //  0.04045
#define GAMMA_INV_THRESHOLD (7827.0f / 2500000.0f)  //  0.0031308
#define GAMMA_LOW_SCALE (323.0f / 25.0f)            // 12.92
#define GAMMA_POWER (12.0f / 5.0f)                  //  2.4
#define GAMMA_XSHIFT (11.0f / 200.0f)               //  0.055

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1092
#define THRESHOLD_6_29TH (6.0f / 29.0f)
#define THRESHOLD_CUBED (powf(THRESHOLD_6_29TH, 3.0))  // (6/29)^3
#define OFFSET_4_29TH (4.0f / 29.0f)
#define SLOPE_THRESHOLD (powf(1.0f / THRESHOLD_6_29TH, 2.0f) / 3.0f)  // (29/6)^2 / 3
#define SLOPE_LAB (3.0f * powf(THRESHOLD_6_29TH, 2.0))                // 3 * (6/29)^2

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1017
#define LTHRESHOLD (216.0f / 24389.0f)  // 0.008856
#define LSCALE (841.0f / 108.0f)        // 7.787
#define LBIAS (16.0f / 116.0f)          // 0.13793103448275862

// -------------------------------------------------------------------------------------
// Helper functions for RGB ↔ LAB conversion (match OpenCV)
// -------------------------------------------------------------------------------------
__device__ float srgb_to_linear(uint8_t c) {
  // OpenCV's gamma correction, input is 8-bit (0-255)
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1023
  float cf = c / 255.0f;
  return (cf <= GAMMA_THRESHOLD) ? cf / GAMMA_LOW_SCALE :
                                   powf((cf + GAMMA_XSHIFT) / (1.0f + GAMMA_XSHIFT), GAMMA_POWER);
}

__device__ float linear_to_srgb(float c) {
  // OpenCV's inverse gamma correction
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1033
  return (c <= GAMMA_INV_THRESHOLD) ?
             GAMMA_LOW_SCALE * c :
             powf(c, 1.0f / GAMMA_POWER) * (1.0 + GAMMA_XSHIFT) - GAMMA_XSHIFT;
}

__device__ float xyz_to_lab_f(float t) {
  // OpenCV-compatible.
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1184
  return (t > LTHRESHOLD) ? cbrtf(t) : (LSCALE * t + LBIAS);
}

__device__ float lab_f_to_xyz(float u) {
  // Inverse: OpenCV-compatible.
  return (u > THRESHOLD_6_29TH) ? (u * u * u) : (SLOPE_LAB * (u - OFFSET_4_29TH));
}

__device__ void rgb_to_lab(uint8_t r, uint8_t g, uint8_t b, float *L, float *a_out, float *b_out) {
  // sRGB to linear RGB (OpenCV expects 8-bit input)
  float rf = srgb_to_linear(r);
  float gf = srgb_to_linear(g);
  float bf = srgb_to_linear(b);

  // Linear RGB to XYZ using OpenCV's  matrix (sRGB D65)
  float x = CV_RGB_XR * rf + CV_RGB_XG * gf + CV_RGB_XB * bf;
  float y = CV_RGB_YR * rf + CV_RGB_YG * gf + CV_RGB_YB * bf;
  float z = CV_RGB_ZR * rf + CV_RGB_ZG * gf + CV_RGB_ZB * bf;

  // Normalize by D65 white point (OpenCV values)
  x = x / D65_WHITE_X;
  y = y / D65_WHITE_Y;
  z = z / D65_WHITE_Z;

  // XYZ to LAB
  float fx = xyz_to_lab_f(x);
  float fy = xyz_to_lab_f(y);
  float fz = xyz_to_lab_f(z);

  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1204
  *L = 116.0f * fy - 16.0f;
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1189
  *a_out = 500.0f * (fx - fy);
  *b_out = 200.0f * (fy - fz);
}

__device__ void lab_to_rgb(float L, float a, float b, uint8_t *r, uint8_t *g, uint8_t *b_out) {
  // LAB to XYZ
  float fy = (L + 16.0f) / 116.0f;
  float fx = a / 500.0f + fy;
  float fz = fy - b / 200.0f;

  // Convert using OpenCV's  D65 white point values
  float x = lab_f_to_xyz(fx) * D65_WHITE_X;
  float y = lab_f_to_xyz(fy) * D65_WHITE_Y;
  float z = lab_f_to_xyz(fz) * D65_WHITE_Z;

  // XYZ to linear RGB using OpenCV's  inverse matrix
  float rf = CV_LAB_XR * x + CV_LAB_XG * y + CV_LAB_XB * z;
  float gf = CV_LAB_YR * x + CV_LAB_YG * y + CV_LAB_YB * z;
  float bf = CV_LAB_ZR * x + CV_LAB_ZG * y + CV_LAB_ZB * z;

  // Linear RGB to sRGB
  rf = linear_to_srgb(rf);
  gf = linear_to_srgb(gf);
  bf = linear_to_srgb(bf);

  // Clamp and convert to uint8 (OpenCV uses rounding)
  *r = (uint8_t)lrintf(dali::clamp(rf * 255.0f, 0.f, 255.f));
  *g = (uint8_t)lrintf(dali::clamp(gf * 255.0f, 0.f, 255.f));
  *b_out = (uint8_t)lrintf(dali::clamp(bf * 255.0f, 0.f, 255.f));
}

// -------------------------------------------------------------------------------------
// Kernel 1: RGB -> LAB L* (uint8). NHWC input (uint8), L* in [0..255] as uint8.
// Uses OpenCV-compatible LAB conversion for consistency with OpenCV CLAHE
// -------------------------------------------------------------------------------------
__global__ void rgb_to_y_u8_nhwc_kernel(const uint8_t *__restrict__ rgb,
                                        uint8_t *__restrict__ y_out, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N) {
    return;
  }

  int c0 = 3 * idx;
  uint8_t r = rgb[c0 + 0];
  uint8_t g = rgb[c0 + 1];
  uint8_t b = rgb[c0 + 2];

  // Convert to LAB L* to match OpenCV CLAHE behavior
  float L, a, b_lab;
  rgb_to_lab(r, g, b, &L, &a, &b_lab);

  // Scale L [0,100] to [0,255] for consistency
  uint8_t L_u8 = (uint8_t)lrintf(dali::clamp(L * 255.0f / 100.0f, 0.f, 255.f));
  y_out[idx] = L_u8;
}

// Vectorized version for better memory bandwidth (processes 4 pixels at once)
__global__ void rgb_to_y_u8_nhwc_vectorized_kernel(const uint8_t *__restrict__ rgb,
                                                   uint8_t *__restrict__ y_out, int H, int W) {
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int N = H * W;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = base_idx + i;
    if (idx >= N) {
      return;
    }

    int c0 = 3 * idx;

    uint8_t r = rgb[c0 + 0];
    uint8_t g = rgb[c0 + 1];
    uint8_t b = rgb[c0 + 2];

    float L, a, b_lab;
    rgb_to_lab(r, g, b, &L, &a, &b_lab);

    uint8_t L_u8 = (uint8_t)lrintf(dali::clamp(L * 255.0f / 100.0f, 0.f, 255.f));
    y_out[idx] = L_u8;
  }
}

void LaunchRGBToYUint8NHWC(const uint8_t *in_rgb, uint8_t *y_plane, int H, int W,
                           cudaStream_t stream) {
  int N = H * W;

  if (N >= 4096) {  // Use vectorized version for larger images
    int threads = 256;
    int blocks = dali::div_ceil(N, threads * 4);  // Each thread processes 4 pixels
    rgb_to_y_u8_nhwc_vectorized_kernel<<<blocks, threads, 0, stream>>>(in_rgb, y_plane, H, W);
  } else {
    int threads = 256;
    int blocks = dali::div_ceil(N, threads);
    rgb_to_y_u8_nhwc_kernel<<<blocks, threads, 0, stream>>>(in_rgb, y_plane, H, W);
  }
}

// -------------------------------------------------------------------------------------
// Fused Kernel: RGB to Y + Histogram per tile (optimized)
// -------------------------------------------------------------------------------------
__global__ void fused_rgb_to_y_hist_kernel(const uint8_t *__restrict__ rgb,
                                           uint8_t *__restrict__ y_out, int H, int W, int tiles_x,
                                           int tiles_y, unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y) {
    return;
  }

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    shist[i] = 0u;
  }
  __syncthreads();

  // Compute tile bounds
  int tile_w = dali::div_ceil(W, tiles_x);
  int tile_h = dali::div_ceil(H, tiles_y);
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);

  // Loop over tile pixels - fused RGB->Y + histogram
  int area = (x1 - x0) * (y1 - y0);
  for (int i = threadIdx.x; i < area; i += blockDim.x) {
    int dy = i / (x1 - x0);
    int dx = i - dy * (x1 - x0);
    int x = x0 + dx;
    int y = y0 + dy;

    int pixel_idx = y * W + x;
    int rgb_idx = 3 * pixel_idx;


    // RGB to LAB L* conversion (match OpenCV ly)
    // Use OpenCV-compatible sRGB to linear conversion (8-bit input)
    uint8_t r = rgb[rgb_idx + 0];
    uint8_t g = rgb[rgb_idx + 1];
    uint8_t b = rgb[rgb_idx + 2];

    float rf = srgb_to_linear(r);
    float gf = srgb_to_linear(g);
    float bf = srgb_to_linear(b);

    // Convert to CIE XYZ using OpenCV's  transformation matrix
    float x_xyz = CV_RGB_XR * rf + CV_RGB_XG * gf + CV_RGB_XB * bf;
    float y_xyz = CV_RGB_YR * rf + CV_RGB_YG * gf + CV_RGB_YB * bf;
    float z_xyz = CV_RGB_ZR * rf + CV_RGB_ZG * gf + CV_RGB_ZB * bf;

    // Normalize by D65 white point (OpenCV  values)
    x_xyz = x_xyz / D65_WHITE_X;
    y_xyz = y_xyz / D65_WHITE_Y;
    z_xyz = z_xyz / D65_WHITE_Z;

    // Convert Y to LAB L* using OpenCV's  threshold and constants
    float fy = (y_xyz > THRESHOLD_CUBED) ? cbrtf(y_xyz) : (SLOPE_THRESHOLD * y_xyz + OFFSET_4_29TH);
    float L = 116.0f * fy - 16.0f;

    // Scale L [0,100] to [0,255] for histogram (OpenCV LAB L* is [0,100])
    uint8_t y_u8 = (uint8_t)lrintf(dali::clamp(L * 255.0f / 100.0f, 0.f, 255.f));  // Store Y value
    y_out[pixel_idx] = y_u8;

    // Add to histogram
    atomicAdd(&shist[static_cast<int>(y_u8)], 1u);
  }
  __syncthreads();

  // Write back histogram to global memory
  unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    g_hist[i] = shist[i];
  }
}

void LaunchFusedRGBToYHist(const uint8_t *rgb, uint8_t *y_plane, int H, int W, int tiles_x,
                           int tiles_y, unsigned int *histograms, cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // Optimized for both compute and shared memory
  size_t shmem = 256 * sizeof(unsigned int);
  fused_rgb_to_y_hist_kernel<<<grid, threads, shmem, stream>>>(rgb, y_plane, H, W, tiles_x, tiles_y,
                                                               histograms);
}

// -------------------------------------------------------------------------------------
// Optimized Kernel: Histograms per tile with warp-privatized reduction (256 bins, uint32)
// -------------------------------------------------------------------------------------
__global__ void hist_per_tile_256_warp_optimized_kernel(const uint8_t *__restrict__ y_plane, int H,
                                                        int W, int tiles_x, int tiles_y,
                                                        unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;
  const int warp_size = 32;
  const int warps_per_block = blockDim.x / warp_size;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y) {
    return;
  }

  int warp_id = threadIdx.x / warp_size;
  int lane_id = threadIdx.x % warp_size;

  // Per-warp private histograms (warps_per_block * 256 bins)
  // This reduces atomic contention significantly
  unsigned int *warp_hist = shist + warp_id * bins;
  unsigned int *global_hist = shist + warps_per_block * bins;  // Final merged histogram

  // Zero per-warp histogram
  for (int i = lane_id; i < bins; i += warp_size) {
    warp_hist[i] = 0u;
  }

  // Zero global histogram (only first warp)
  if (warp_id == 0) {
    for (int i = lane_id; i < bins; i += warp_size) {
      global_hist[i] = 0u;
    }
  }
  __syncthreads();

  // Compute tile bounds
  int tile_w = dali::div_ceil(W, tiles_x);
  int tile_h = dali::div_ceil(H, tiles_y);
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);

  // Each warp processes its portion of the tile
  int area = (x1 - x0) * (y1 - y0);
  for (int i = threadIdx.x; i < area; i += blockDim.x) {
    int dy = i / (x1 - x0);
    int dx = i % (x1 - x0);
    int x = x0 + dx;
    int y = y0 + dy;
    uint8_t v = y_plane[y * W + x];

    // Atomic to warp-private histogram (much less contention)
    atomicAdd(&warp_hist[static_cast<int>(v)], 1u);
  }
  __syncthreads();

  // Merge warp histograms to final histogram
  for (int bin = lane_id; bin < bins; bin += warp_size) {
    unsigned int sum = 0u;
    for (int w = 0; w < warps_per_block; ++w) {
      sum += shist[w * bins + bin];
    }
    global_hist[bin] = sum;
  }
  __syncthreads();

  // Write back to global memory
  unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    g_hist[i] = global_hist[i];
  }
}

void LaunchHistPerTile256WarpOptimized(const uint8_t *y_plane, int H, int W, int tiles_x,
                                       int tiles_y, unsigned int *histograms, cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // 16 warps per block
  int warps_per_block = threads / 32;
  // Shared memory: warps_per_block * 256 (private) + 256 (final)
  size_t shmem = (warps_per_block + 1) * 256 * sizeof(unsigned int);
  hist_per_tile_256_warp_optimized_kernel<<<grid, threads, shmem, stream>>>(y_plane, H, W, tiles_x,
                                                                            tiles_y, histograms);
}

// Original version kept for fallback
__global__ void hist_per_tile_256_kernel(const uint8_t *__restrict__ y_plane, int H, int W,
                                         int tiles_x, int tiles_y,
                                         unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y) {
    return;
  }

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    shist[i] = 0u;
  }
  __syncthreads();

  // Compute tile bounds
  int tile_w = dali::div_ceil(W, tiles_x);
  int tile_h = dali::div_ceil(H, tiles_y);
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);

  // Loop over tile pixels
  int area = (x1 - x0) * (y1 - y0);
  for (int i = threadIdx.x; i < area; i += blockDim.x) {
    int dy = i / (x1 - x0);
    int dx = i % (x1 - x0);
    int x = x0 + dx;
    int y = y0 + dy;
    uint8_t v = y_plane[y * W + x];
    atomicAdd(&shist[static_cast<int>(v)], 1u);
  }
  __syncthreads();

  // Write back to global memory
  unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    g_hist[i] = shist[i];
  }
}

void LaunchHistPerTile256(const uint8_t *y_plane, int H, int W, int tiles_x, int tiles_y,
                          unsigned int *histograms, cudaStream_t stream) {
  // Use warp-optimized version for larger tiles (where contention is higher)
  int tile_area = dali::div_ceil(W, tiles_x) * dali::div_ceil(H, tiles_y);
  if (tile_area >= 1024) {  // Threshold where warp optimization pays off
    LaunchHistPerTile256WarpOptimized(y_plane, H, W, tiles_x, tiles_y, histograms, stream);
  } else {
    // Use original version for small tiles
    dim3 grid(tiles_x, tiles_y, 1);
    int threads = 512;
    size_t shmem = 256 * sizeof(unsigned int);
    hist_per_tile_256_kernel<<<grid, threads, shmem, stream>>>(y_plane, H, W, tiles_x, tiles_y,
                                                               histograms);
  }
}

// -------------------------------------------------------------------------------------
// Kernel 3: Clip + CDF -> LUT per tile (uint8 LUT).
// -------------------------------------------------------------------------------------
__global__ void clip_cdf_lut_256_kernel(unsigned int *__restrict__ histograms, int tiles_x,
                                        int tiles_y, int tile_w,
                                        int tile_h,  // nominal, last tiles smaller
                                        int W, int H, float clip_limit_rel,
                                        uint8_t *__restrict__ luts) {
  const int bins = 256;
  int tid = threadIdx.x;

  int tx = blockIdx.x;
  int ty = blockIdx.y;
  if (tx >= tiles_x || ty >= tiles_y) {
    return;
  }

  // Actual tile bounds (handle edges)
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);
  int area = max(1, (x1 - x0) * (y1 - y0));

  unsigned int *hist = histograms + (ty * tiles_x + tx) * bins;
  __shared__ unsigned int h[256];
  __shared__ unsigned int cdf[256];

  // Load histogram
  for (int i = tid; i < bins; i += blockDim.x) {
    h[i] = hist[i];
  }
  __syncthreads();

  // Compute clip limit (match OpenCV ly)
  float clip_limit_f = clip_limit_rel * area / bins;
  int limit_int = static_cast<int>(clip_limit_f);
  int limit = max(limit_int, 1);
  unsigned int limit_u = static_cast<unsigned int>(limit);

  // Clip and accumulate excess
  __shared__ unsigned int excess;
  if (tid == 0) {
    excess = 0u;
  }
  __syncthreads();

  for (int i = tid; i < bins; i += blockDim.x) {
    unsigned int v = h[i];
    if (v > limit_u) {
      unsigned int over = v - limit_u;
      h[i] = limit_u;
      atomicAdd(&excess, over);
    }
  }
  __syncthreads();

  // Redistribute excess using OpenCV's  algorithm
  unsigned int redistBatch = excess / bins;  // OpenCV: redistBatch = clipped / histSize
  unsigned int residual = excess % bins;     // OpenCV: residual = clipped - redistBatch * histSize

  for (int i = tid; i < bins; i += blockDim.x) {
    h[i] += redistBatch;
  }
  __syncthreads();

  // Distribute residual using OpenCV's  step pattern
  if (tid == 0 && residual > 0) {
    unsigned int residualStep = max(bins / residual, 1u);  // OpenCV: MAX(histSize / residual, 1)
    for (unsigned int i = 0; i < bins && residual > 0; i += residualStep, residual--) {
      h[i]++;  // OpenCV: tileHist[i]++
    }
  }
  __syncthreads();

  // Prefix-sum (CDF)
  if (tid == 0) {
    unsigned int acc = 0u;
    for (int i = 0; i < bins; ++i) {
      acc += h[i];
      cdf[i] = acc;
    }
  }
  __syncthreads();

  // Build LUT using OpenCV's  scaling methodology
  uint8_t *lut = luts + (ty * tiles_x + tx) * bins;

  // OpenCV uses: lutScale = (histSize - 1) / tileSizeTotal
  float lutScale = static_cast<float>(bins - 1) / static_cast<float>(area);

  for (int i = tid; i < bins; i += blockDim.x) {
    // OpenCV applies: lut[i] = saturate_cast<uchar>(sum * lutScale + 0.5f)
    float val = static_cast<float>(cdf[i]) * lutScale + 0.5f;
    lut[i] = static_cast<uint8_t>(dali::clamp(val, 0.f, 255.f));
  }
}

void LaunchClipCdfToLut256(unsigned int *histograms, int H, int W, int tiles_x, int tiles_y,
                           float clip_limit_rel, uint8_t *luts, cudaStream_t stream) {
  int tile_w = dali::div_ceil(W, tiles_x);
  int tile_h = dali::div_ceil(H, tiles_y);
  dim3 grid(tiles_x, tiles_y, 1);

  // 256 threads allows more blocks per SM, improving overall throughput
  int threads = 256;
  clip_cdf_lut_256_kernel<<<grid, threads, 0, stream>>>(histograms, tiles_x, tiles_y, tile_w,
                                                        tile_h, W, H, clip_limit_rel, luts);
}

// -------------------------------------------------------------------------------------
// Apply LUT (GRAYSCALE) — vectorized/original/optimized; OpenCV rounding
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_gray_vectorized_kernel(uint8_t *__restrict__ dst_y,
                                                          const uint8_t *__restrict__ src_y, int H,
                                                          int W, int tiles_x, int tiles_y,
                                                          const uint8_t *__restrict__ luts) {
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int N = H * W;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = base_idx + i;
    if (idx >= N) {
      return;
    }

    int y = idx / W;
    int x = idx - y * W;

    // Tile geometry - match OpenCV ly (same as RGB version)
    float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
    float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);

    float gx = x * inv_tw - 0.5f;  // OpenCV: x * inv_tw - 0.5f
    float gy = y * inv_th - 0.5f;  // OpenCV: y * inv_th - 0.5f
    int tx = static_cast<int>(floorf(gx));
    int ty = static_cast<int>(floorf(gy));
    float fx = gx - tx;
    float fy = gy - ty;

    // Handle border cases properly
    int tx0, ty0, tx1, ty1;

    if (tx < 0) {
      tx0 = tx1 = 0;
      fx = 0.f;
    } else if (tx >= tiles_x - 1) {
      tx0 = tx1 = tiles_x - 1;
      fx = 0.f;
    } else {
      tx0 = tx;
      tx1 = tx + 1;
      fx = dali::clamp(fx, 0.f, 1.f);
    }

    if (ty < 0) {
      ty0 = ty1 = 0;
      fy = 0.f;
    } else if (ty >= tiles_y - 1) {
      ty0 = ty1 = tiles_y - 1;
      fy = 0.f;
    } else {
      ty0 = ty;
      ty1 = ty + 1;
      fy = dali::clamp(fy, 0.f, 1.f);
    }

    int bins = 256;
    const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
    const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
    const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
    const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

    uint8_t v = src_y[idx];
    float v_tl = lut_tl[v];
    float v_tr = lut_tr[v];
    float v_bl = lut_bl[v];
    float v_br = lut_br[v];

    // Bilinear blend
    float v_top = v_tl * (1.f - fx) + v_tr * fx;
    float v_bot = v_bl * (1.f - fx) + v_br * fx;
    float v_out = v_top * (1.f - fy) + v_bot * fy;

    int outi = static_cast<int>(lrintf(dali::clamp(v_out, 0.f, 255.f)));
    dst_y[idx] = (uint8_t)outi;
  }
}

// Original single-pixel version
__global__ void apply_lut_bilinear_gray_kernel(uint8_t *__restrict__ dst_y,
                                               const uint8_t *__restrict__ src_y, int H, int W,
                                               int tiles_x, int tiles_y,
                                               const uint8_t *__restrict__ luts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N) {
    return;
  }

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - match OpenCV ly (same as RGB version)
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);  // 1.0f / tileSize.width
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);  // 1.0f / tileSize.height

  // Tile coordinates (match OpenCV ly)
  float gx = x * inv_tw - 0.5f;           // OpenCV: x * inv_tw - 0.5f
  float gy = y * inv_th - 0.5f;           // OpenCV: y * inv_th - 0.5f
  int tx = static_cast<int>(floorf(gx));  // OpenCV: cvFloor(txf)
  int ty = static_cast<int>(floorf(gy));  // OpenCV: cvFloor(tyf)
  float fx = gx - tx;                     // OpenCV: xa = txf - tx1
  float fy = gy - ty;                     // OpenCV: ya = tyf - ty1

  // Handle border cases properly
  int tx0, ty0, tx1, ty1;

  if (tx < 0) {
    tx0 = tx1 = 0;
    fx = 0.f;
  } else if (tx >= tiles_x - 1) {
    tx0 = tx1 = tiles_x - 1;
    fx = 0.f;
  } else {
    tx0 = tx;
    tx1 = tx + 1;
    fx = dali::clamp(fx, 0.f, 1.f);
  }

  if (ty < 0) {
    ty0 = ty1 = 0;
    fy = 0.f;
  } else if (ty >= tiles_y - 1) {
    ty0 = ty1 = tiles_y - 1;
    fy = 0.f;
  } else {
    ty0 = ty;
    ty1 = ty + 1;
    fy = dali::clamp(fy, 0.f, 1.f);
  }

  int bins = 256;
  const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
  const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
  const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
  const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

  uint8_t v = src_y[idx];
  float v_tl = lut_tl[v];
  float v_tr = lut_tr[v];
  float v_bl = lut_bl[v];
  float v_br = lut_br[v];

  // Bilinear blend
  float v_top = v_tl * (1.f - fx) + v_tr * fx;
  float v_bot = v_bl * (1.f - fx) + v_br * fx;
  float v_out = v_top * (1.f - fy) + v_bot * fy;

  int outi = static_cast<int>(lrintf(dali::clamp(v_out, 0.f, 255.f)));
  dst_y[idx] = (uint8_t)outi;
}

// ---------------------------
// Optimized Kernel: Apply LUT
// ---------------------------
__global__ void apply_lut_bilinear_gray_optimized_kernel(uint8_t *__restrict__ dst_y,
                                                         const uint8_t *__restrict__ src_y, int H,
                                                         int W, int tiles_x, int tiles_y,
                                                         const uint8_t *__restrict__ luts,
                                                         int bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N) {
    return;
  }

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - match OpenCV ly
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);

  // Tile coordinates (match OpenCV ly)
  float gx = x * inv_tw - 0.5f;
  float gy = y * inv_th - 0.5f;
  int tx = static_cast<int>(floorf(gx));
  int ty = static_cast<int>(floorf(gy));
  float fx = gx - tx;
  float fy = gy - ty;

  // Handle border cases
  int tx0, ty0, tx1, ty1;

  if (tx < 0) {
    tx0 = tx1 = 0;
    fx = 0.f;
  } else if (tx >= tiles_x - 1) {
    tx0 = tx1 = tiles_x - 1;
    fx = 0.f;
  } else {
    tx0 = tx;
    tx1 = tx + 1;
    fx = dali::clamp(fx, 0.f, 1.f);
  }

  if (ty < 0) {
    ty0 = ty1 = 0;
    fy = 0.f;
  } else if (ty >= tiles_y - 1) {
    ty0 = ty1 = tiles_y - 1;
    fy = 0.f;
  } else {
    ty0 = ty;
    ty1 = ty + 1;
    fy = dali::clamp(fy, 0.f, 1.f);
  }

  uint8_t v = src_y[idx];

  // Use regular memory access for LUT lookups
  const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
  const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
  const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
  const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

  float v_tl = lut_tl[v];
  float v_tr = lut_tr[v];
  float v_bl = lut_bl[v];
  float v_br = lut_br[v];

  // Bilinear blend
  float v_top = v_tl * (1.f - fx) + v_tr * fx;
  float v_bot = v_bl * (1.f - fx) + v_br * fx;
  float v_out = v_top * (1.f - fy) + v_bot * fy;

  int outi = static_cast<int>(lrintf(dali::clamp(v_out, 0.f, 255.f)));
  dst_y[idx] = (uint8_t)outi;
}

void LaunchApplyLUTBilinearToGrayOptimized(uint8_t *dst_gray, const uint8_t *src_gray, int H, int W,
                                           int tiles_x, int tiles_y, const uint8_t *luts,
                                           cudaStream_t stream) {
  int N = H * W;
  int threads = 256;
  int blocks = dali::div_ceil(N, threads);
  apply_lut_bilinear_gray_optimized_kernel<<<blocks, threads, 0, stream>>>(
      dst_gray, src_gray, H, W, tiles_x, tiles_y, luts, 256);
}

// Update the main launcher to use optimized version
void LaunchApplyLUTBilinearToGray(uint8_t *dst_gray, const uint8_t *src_gray, int H, int W,
                                  int tiles_x, int tiles_y, const uint8_t *luts,
                                  cudaStream_t stream) {
  int N = H * W;
  int total_tiles = tiles_x * tiles_y;

  // Use optimized version for larger tile counts where better performance is needed
  if (total_tiles >= 32 && N >= 16384) {
    LaunchApplyLUTBilinearToGrayOptimized(dst_gray, src_gray, H, W, tiles_x, tiles_y, luts, stream);
  } else if (N >= 8192) {  // Use vectorized version for medium images
    int threads = 256;
    int blocks = dali::div_ceil(N, threads * 4);
    apply_lut_bilinear_gray_vectorized_kernel<<<blocks, threads, 0, stream>>>(
        dst_gray, src_gray, H, W, tiles_x, tiles_y, luts);
  } else {
    // Use original version for smaller images
    int threads = 512;
    int blocks = dali::div_ceil(N, threads);
    apply_lut_bilinear_gray_kernel<<<blocks, threads, 0, stream>>>(dst_gray, src_gray, H, W,
                                                                   tiles_x, tiles_y, luts);
  }
}

// -------------------------------------------------------------------------------------
// Optimized Vectorized Kernel: Apply LUT for RGB using vectorized memory access
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_rgb_vectorized_kernel(uint8_t *__restrict__ dst_rgb,
                                                         const uint8_t *__restrict__ src_rgb,
                                                         const uint8_t *__restrict__ src_y, int H,
                                                         int W, int tiles_x, int tiles_y,
                                                         const uint8_t *__restrict__ luts) {
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // Process 2 pixels per thread
  int N = H * W;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int idx = base_idx + i;
    if (idx >= N) {
      return;
    }

    int y = idx / W;
    int x = idx - y * W;

    // --- Tile geometry and interpolation (OpenCV-style fractional indices) ---
    float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
    float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);
    float gx = x * inv_tw - 0.5f;
    float gy = y * inv_th - 0.5f;
    int tx = static_cast<int>(floorf(gx));
    int ty = static_cast<int>(floorf(gy));
    float fx = gx - tx;
    float fy = gy - ty;

    // REPLICATE border policy (match gray/OpenCV)
    int tx0, tx1, ty0, ty1;
    if (tx < 0) {
      tx0 = tx1 = 0;
      fx = 0.f;
    } else if (tx >= tiles_x - 1) {
      tx0 = tx1 = tiles_x - 1;
      fx = 0.f;
    } else {
      tx0 = tx;
      tx1 = tx + 1;
      fx = dali::clamp(fx, 0.f, 1.f);
    }

    if (ty < 0) {
      ty0 = ty1 = 0;
      fy = 0.f;
    } else if (ty >= tiles_y - 1) {
      ty0 = ty1 = tiles_y - 1;
      fy = 0.f;
    } else {
      ty0 = ty;
      ty1 = ty + 1;
      fy = dali::clamp(fy, 0.f, 1.f);
    }
    // --- End tile geometry fix ---

    int bins = 256;
    const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
    const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
    const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
    const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

    uint8_t orig_L_u8 = src_y[idx];
    float v_tl = lut_tl[orig_L_u8];
    float v_tr = lut_tr[orig_L_u8];
    float v_bl = lut_bl[orig_L_u8];
    float v_br = lut_br[orig_L_u8];

    float v_top = v_tl * (1.f - fx) + v_tr * fx;
    float v_bot = v_bl * (1.f - fx) + v_br * fx;
    float enhanced_L_u8 = v_top * (1.f - fy) + v_bot * fy;

    // Convert original RGB to LAB
    int base = 3 * idx;
    uint8_t orig_r = src_rgb[base + 0];
    uint8_t orig_g = src_rgb[base + 1];
    uint8_t orig_b = src_rgb[base + 2];

    float orig_L, orig_a, orig_b_lab;
    rgb_to_lab(orig_r, orig_g, orig_b, &orig_L, &orig_a, &orig_b_lab);

    // Replace L* with enhanced version, keep a* and b* unchanged
    float enhanced_L =
        dali::clamp(static_cast<float>(lrintf(enhanced_L_u8 * 100.0f / 255.0f)), 0.0f, 100.0f);

    // Convert LAB back to RGB
    uint8_t new_r, new_g, new_b;
    lab_to_rgb(enhanced_L, orig_a, orig_b_lab, &new_r, &new_g, &new_b);

    dst_rgb[base + 0] = new_r;
    dst_rgb[base + 1] = new_g;
    dst_rgb[base + 2] = new_b;
  }
}

// Original single-pixel RGB version
__global__ void apply_lut_bilinear_rgb_kernel(uint8_t *__restrict__ dst_rgb,
                                              const uint8_t *__restrict__ src_rgb,
                                              const uint8_t *__restrict__ src_y,  // original L*
                                              int H, int W, int tiles_x, int tiles_y,
                                              const uint8_t *__restrict__ luts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N) {
    return;
  }

  int y = idx / W;
  int x = idx - y * W;

  // --- Tile geometry and interpolation (OpenCV-style fractional indices) ---
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);
  float gx = x * inv_tw - 0.5f;
  float gy = y * inv_th - 0.5f;
  int tx = static_cast<int>(floorf(gx));
  int ty = static_cast<int>(floorf(gy));
  float fx = gx - tx;
  float fy = gy - ty;

  // REPLICATE border policy (match gray/OpenCV)
  int tx0, tx1, ty0, ty1;
  if (tx < 0) {
    tx0 = tx1 = 0;
    fx = 0.f;
  } else if (tx >= tiles_x - 1) {
    tx0 = tx1 = tiles_x - 1;
    fx = 0.f;
  } else {
    tx0 = tx;
    tx1 = tx + 1;
    fx = dali::clamp(fx, 0.f, 1.f);
  }

  if (ty < 0) {
    ty0 = ty1 = 0;
    fy = 0.f;
  } else if (ty >= tiles_y - 1) {
    ty0 = ty1 = tiles_y - 1;
    fy = 0.f;
  } else {
    ty0 = ty;
    ty1 = ty + 1;
    fy = dali::clamp(fy, 0.f, 1.f);
  }
  // --- End tile geometry fix ---

  int bins = 256;
  const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
  const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
  const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
  const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

  uint8_t orig_L_u8 = src_y[idx];  // Original L* value scaled [0,255]
  float v_tl = lut_tl[orig_L_u8];
  float v_tr = lut_tr[orig_L_u8];
  float v_bl = lut_bl[orig_L_u8];
  float v_br = lut_br[orig_L_u8];

  float v_top = v_tl * (1.f - fx) + v_tr * fx;
  float v_bot = v_bl * (1.f - fx) + v_br * fx;
  float enhanced_L_u8 = v_top * (1.f - fy) + v_bot * fy;

  // Convert original RGB to LAB
  int base = 3 * idx;
  uint8_t orig_r = src_rgb[base + 0];
  uint8_t orig_g = src_rgb[base + 1];
  uint8_t orig_b = src_rgb[base + 2];

  float orig_L, orig_a, orig_b_lab;
  rgb_to_lab(orig_r, orig_g, orig_b, &orig_L, &orig_a, &orig_b_lab);

  // Replace L* with enhanced version, keep a* and b* unchanged
  float enhanced_L =
      dali::clamp(static_cast<float>(lrintf(enhanced_L_u8 * 100.0f / 255.0f)), 0.0f, 100.0f);

  // Convert LAB back to RGB
  uint8_t new_r, new_g, new_b;
  lab_to_rgb(enhanced_L, orig_a, orig_b_lab, &new_r, &new_g, &new_b);

  dst_rgb[base + 0] = new_r;
  dst_rgb[base + 1] = new_g;
  dst_rgb[base + 2] = new_b;
}

void LaunchApplyLUTBilinearToRGB(uint8_t *dst_rgb, const uint8_t *src_rgb, const uint8_t *src_y,
                                 int H, int W, int tiles_x, int tiles_y, const uint8_t *luts,
                                 cudaStream_t stream) {
  int N = H * W;

  // Use vectorized version for larger images
  if (N >= 8192) {                                // Threshold for vectorized processing
    int threads = 256;                            // Better occupancy with complex RGB processing
    int blocks = dali::div_ceil(N, threads * 2);  // Each thread processes 2 pixels
    apply_lut_bilinear_rgb_vectorized_kernel<<<blocks, threads, 0, stream>>>(
        dst_rgb, src_rgb, src_y, H, W, tiles_x, tiles_y, luts);
  } else {
    // Use original version for smaller images
    int threads = 512;
    int blocks = dali::div_ceil(N, threads);
    apply_lut_bilinear_rgb_kernel<<<blocks, threads, 0, stream>>>(dst_rgb, src_rgb, src_y, H, W,
                                                                  tiles_x, tiles_y, luts);
  }
}

// -------------------------------------------------------------------------------------
// Mega-Fused Kernel: Histogram + Clip + CDF + LUT generation in one pass
// -------------------------------------------------------------------------------------
__global__ void mega_fused_hist_clip_cdf_lut_kernel(const uint8_t *__restrict__ y_plane, int H,
                                                    int W, int tiles_x, int tiles_y, int tile_w,
                                                    int tile_h, float clip_limit_rel,
                                                    uint8_t *__restrict__ luts) {
  extern __shared__ unsigned int sdata[];  // Dynamic shared memory
  const int bins = 256;
  const int warp_size = 32;
  const int warps_per_block = blockDim.x / warp_size;

  // Shared memory layout:
  // [0...warps_per_block*256) = per-warp histograms
  // [warps_per_block*256...warps_per_block*256+256) = final histogram
  // [warps_per_block*256+256...warps_per_block*256+512) = CDF
  unsigned int *warp_hist = sdata;
  unsigned int *hist = sdata + warps_per_block * bins;
  unsigned int *cdf = hist + bins;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y) {
    return;
  }

  int warp_id = threadIdx.x / warp_size;
  int lane_id = threadIdx.x % warp_size;

  // Initialize shared memory
  unsigned int *my_warp_hist = warp_hist + warp_id * bins;
  for (int i = lane_id; i < bins; i += warp_size) {
    my_warp_hist[i] = 0u;
  }

  if (warp_id == 0) {
    for (int i = lane_id; i < bins; i += warp_size) {
      hist[i] = 0u;
      cdf[i] = 0u;
    }
  }
  __syncthreads();

  // Compute actual tile bounds
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);
  int area = max(1, (x1 - x0) * (y1 - y0));

  // Build per-warp histograms
  int tile_area = (x1 - x0) * (y1 - y0);
  for (int i = threadIdx.x; i < tile_area; i += blockDim.x) {
    int dy = i / (x1 - x0);
    int dx = i % (x1 - x0);
    int x = x0 + dx;
    int y = y0 + dy;
    uint8_t v = y_plane[y * W + x];
    atomicAdd(&my_warp_hist[static_cast<int>(v)], 1u);
  }
  __syncthreads();

  // Merge warp histograms
  for (int bin = lane_id; bin < bins; bin += warp_size) {
    unsigned int sum = 0u;
    for (int w = 0; w < warps_per_block; ++w) {
      sum += warp_hist[w * bins + bin];
    }
    hist[bin] = sum;
  }
  __syncthreads();

  // Clip histogram and redistribute excess
  float clip_limit_f = clip_limit_rel * area / bins;
  unsigned int limit = max(static_cast<unsigned int>(clip_limit_f), 1u);

  __shared__ unsigned int excess;
  if (threadIdx.x == 0) {
    excess = 0u;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    unsigned int v = hist[i];
    if (v > limit) {
      unsigned int over = v - limit;
      hist[i] = limit;
      atomicAdd(&excess, over);
    }
  }
  __syncthreads();

  // Redistribute excess
  unsigned int redistBatch = excess / bins;
  unsigned int residual = excess % bins;

  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    hist[i] += redistBatch;
  }
  __syncthreads();

  // Distribute residual (OpenCV: one-by-one, bin order)
  if (threadIdx.x == 0 && residual > 0) {
    for (unsigned int i = 0; i < bins && residual > 0; ++i) {
      hist[i]++;
      --residual;
    }
  }
  __syncthreads();

  // Compute CDF (prefix sum)
  if (threadIdx.x == 0) {
    unsigned int acc = 0u;
    for (int i = 0; i < bins; ++i) {
      acc += hist[i];
      cdf[i] = acc;
    }
  }
  __syncthreads();

  // Generate LUT with proper rounding
  uint8_t *lut = luts + (ty * tiles_x + tx) * bins;
  float lutScale = static_cast<float>(bins - 1) / static_cast<float>(area);

  for (int i = threadIdx.x; i < bins; i += blockDim.x) {
    float val = static_cast<float>(cdf[i]) * lutScale + 0.5f;  // OpenCV rounding
    lut[i] = static_cast<uint8_t>(dali::clamp(val, 0.f, 255.f));
  }
}

void LaunchMegaFusedHistClipCdfLut(const uint8_t *y_plane, int H, int W, int tiles_x, int tiles_y,
                                   float clip_limit_rel, uint8_t *luts, cudaStream_t stream) {
  int tile_w = dali::div_ceil(W, tiles_x);
  int tile_h = dali::div_ceil(H, tiles_y);
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 256;  // Optimized for occupancy

  // Shared memory: warp_hists + hist + cdf
  int warps_per_block = threads / 32;
  size_t shmem = (warps_per_block + 2) * 256 * sizeof(unsigned int);

  mega_fused_hist_clip_cdf_lut_kernel<<<grid, threads, shmem, stream>>>(
      y_plane, H, W, tiles_x, tiles_y, tile_w, tile_h, clip_limit_rel, luts);
}

namespace dali {

void LaunchCLAHE_Grayscale_U8_NHWC(uint8_t *dst_gray, const uint8_t *src_gray, int H, int W,
                                   int tiles_x, int tiles_y, float clip_limit_rel,
                                   unsigned int *tmp_histograms,  // tiles*bins
                                   uint8_t *tmp_luts,             // tiles*bins
                                   cudaStream_t stream) {
  // Use mega-fused version for larger images where the fusion overhead pays off
  int total_tiles = tiles_x * tiles_y;
  if (total_tiles >= 16) {  // Threshold where fusion is beneficial
    LaunchMegaFusedHistClipCdfLut(src_gray, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts,
                                  stream);
  } else {
    // Use traditional 3-kernel approach for smaller tile counts
    LaunchHistPerTile256(src_gray, H, W, tiles_x, tiles_y, tmp_histograms, stream);
    LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  }
  LaunchApplyLUTBilinearToGray(dst_gray, src_gray, H, W, tiles_x, tiles_y, tmp_luts, stream);
  CUDA_CALL(cudaGetLastError());
}

void LaunchCLAHE_RGB_U8_NHWC(uint8_t *dst_rgb, const uint8_t *src_rgb,
                             uint8_t *y_plane,  // [H*W]
                             int H, int W, int tiles_x, int tiles_y, float clip_limit_rel,
                             unsigned int *tmp_histograms,  // tiles*bins
                             uint8_t *tmp_luts,             // tiles*bins
                             cudaStream_t stream) {
  LaunchRGBToYUint8NHWC(src_rgb, y_plane, H, W, stream);
  LaunchHistPerTile256(y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
  LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  LaunchApplyLUTBilinearToRGB(dst_rgb, src_rgb, y_plane, H, W, tiles_x, tiles_y, tmp_luts, stream);
  CUDA_CALL(cudaGetLastError());
}

// Optimized version using fused RGB->Y + histogram kernel
void LaunchCLAHE_RGB_U8_NHWC_Optimized(uint8_t *dst_rgb, const uint8_t *src_rgb,
                                       uint8_t *y_plane,  // [H*W]
                                       int H, int W, int tiles_x, int tiles_y, float clip_limit_rel,
                                       unsigned int *tmp_histograms,  // tiles*bins
                                       uint8_t *tmp_luts,             // tiles*bins
                                       cudaStream_t stream) {
  // Fused RGB->Y conversion + histogram computation (saves one kernel launch + memory round-trip)
  LaunchFusedRGBToYHist(src_rgb, y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
  LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  LaunchApplyLUTBilinearToRGB(dst_rgb, src_rgb, y_plane, H, W, tiles_x, tiles_y, tmp_luts, stream);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace dali
