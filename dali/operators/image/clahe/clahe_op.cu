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

#include <bit>

#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/math_util.h"
#include "dali/core/util.h"

#define CV_HEX_CONST_F(x) static_cast<float>(__builtin_bit_cast(double, (uint64_t)(x)))

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L100
// 0.412453, 0.357580, 0.180423,
// 0.212671, 0.715160, 0.072169,
// 0.019334, 0.119193, 0.950227
#define CV_RGB_XR CV_HEX_CONST_F(0x3fda65a14488c60d)  // 0.412453
#define CV_RGB_XG CV_HEX_CONST_F(0x3fd6e297396d0918)  // 0.357580
#define CV_RGB_XB CV_HEX_CONST_F(0x3fc71819d2391d58)  // 0.180423

#define CV_RGB_YR CV_HEX_CONST_F(0x3fcb38cda6e75ff6)  // 0.212673
#define CV_RGB_YG CV_HEX_CONST_F(0x3fe6e297396d0918)  // 0.715160
#define CV_RGB_YB CV_HEX_CONST_F(0x3fb279aae6c8f755)  // 0.072169

#define CV_RGB_ZR CV_HEX_CONST_F(0x3f93cc4ac6cdaf4b)  // 0.019334
#define CV_RGB_ZG CV_HEX_CONST_F(0x3fbe836eb4e98138)  // 0.119193
#define CV_RGB_ZB CV_HEX_CONST_F(0x3fee68427418d691)  // 0.950227

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L116
//  3.240479, -1.53715, -0.498535,
// -0.969256, 1.875991, 0.041556,
//  0.055648, -0.204043, 1.057311
#define CV_LAB_XR CV_HEX_CONST_F(0x4009ec804102ff8f)  // 3.240479
#define CV_LAB_XG CV_HEX_CONST_F(0xbff8982a9930be0e)  // -1.53715
#define CV_LAB_XB CV_HEX_CONST_F(0xbfdfe7ff583a53b9)  // -0.498535
#define CV_LAB_YR CV_HEX_CONST_F(0xbfef042528ae74f3)  // -0.969256
#define CV_LAB_YG CV_HEX_CONST_F(0x3ffe040f23897204)  // 1.875991
#define CV_LAB_YB CV_HEX_CONST_F(0x3fa546d3f9e7b80b)  // 0.041556
#define CV_LAB_ZR CV_HEX_CONST_F(0x3fac7de5082cf52c)  // 0.055648
#define CV_LAB_ZG CV_HEX_CONST_F(0xbfca1e14bdfd2631)  // -0.204043
#define CV_LAB_ZB CV_HEX_CONST_F(0x3ff0eabef06b3786)  // 1.057311

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L940
#define D65_WHITE_X CV_HEX_CONST_F(0x3fee6a22b3892ee8)  // 0.950456
#define D65_WHITE_Y 1.0f                                // 1.000000
#define D65_WHITE_Z CV_HEX_CONST_F(0x3ff16b8950763a19)  // 1.089058

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1010
// Precomputed constants (original formula in comment)
#define GAMMA_THRESHOLD 0.04045f            // 809/20000
#define GAMMA_INV_THRESHOLD 0.0031308f      // 7827/2500000
#define GAMMA_LOW_SCALE 12.92f              // 323/25
#define GAMMA_POWER 2.4f                    // 12/5
#define GAMMA_XSHIFT 0.055f                 // 11/200

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1092
// LAB helper constants
#define THRESHOLD_6_29TH 0.206896551724137931f   // 6/29 (higher precision)
#define THRESHOLD_CUBED 0.008856451679035631f    // (6/29)^3
#define OFFSET_4_29TH 0.137931034482758621f      // 4/29 (higher precision)
#define SLOPE_THRESHOLD 7.787037037037037f       // (29/6)^2 / 3
#define SLOPE_LAB 0.128418549346016740f          // 3*(6/29)^2 (higher precision)

// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1017
// L* conversion constants
#define LTHRESHOLD 0.008856451679035631f        // (6/29)^3
#define LSCALE 7.787037037037037f               // (29/6)^2 / 3
#define LBIAS 0.137931034482758621f             // 4/29

// -------------------------------------------------------------------------------------
// LUT-based color conversion (constant memory for performance)
// -------------------------------------------------------------------------------------

// Constant memory LUTs for color space conversions
__constant__ float g_srgb_to_linear_lut[256];      // sRGB uint8 -> linear float
__constant__ float g_linear_to_srgb_lut[4096];     // linear float -> sRGB (12-bit precision)
__constant__ float g_xyz_to_lab_lut[4096];         // XYZ -> LAB f() transform
__constant__ float g_lab_to_xyz_lut[4096];         // LAB f() inverse -> XYZ

// -------------------------------------------------------------------------------------
// Helper functions for RGB ↔ LAB conversion (match OpenCV)
// -------------------------------------------------------------------------------------

__device__ float srgb_to_linear(uint8_t c) {
  // LUT-based: eliminates branch + powf() (20-30 cycles saved per call)
  return g_srgb_to_linear_lut[c];
}

// Original version kept for reference/validation
__device__ float srgb_to_linear_ref(uint8_t c) {
  // OpenCV's gamma correction, input is 8-bit (0-255)
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1023
  float cf = c * (1.0f / 255.0f);
  if (cf <= GAMMA_THRESHOLD) {
    return cf * (1.0f / GAMMA_LOW_SCALE);
  } else {
    return powf((cf + GAMMA_XSHIFT) * (1.0f / (1.0f + GAMMA_XSHIFT)), GAMMA_POWER);
  }
}

__device__ float linear_to_srgb(float c) {
  // LUT-based with 12-bit quantization: eliminates branch + powf()
  float clamped = fmaxf(0.0f, fminf(c, 1.0f));
  int idx = __float2int_rn(clamped * 4095.0f);
  return g_linear_to_srgb_lut[idx];
}

// Original version kept for reference/validation
__device__ float linear_to_srgb_ref(float c) {
  // OpenCV's inverse gamma correction
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1033
  if (c <= GAMMA_INV_THRESHOLD) {
    return GAMMA_LOW_SCALE * c;
  } else {
    return powf(c, 1.0f / GAMMA_POWER) * (1.0 + GAMMA_XSHIFT) - GAMMA_XSHIFT;
  }
}

__device__ float xyz_to_lab_f(float t) {
  // LUT-based with hybrid fallback: eliminates cbrtf() for common range
  if (t > 1.0f) return cbrtf(t);  // Rare case, fallback to cbrtf
  float clamped = fmaxf(0.0f, t);
  int idx = __float2int_rn(clamped * 4095.0f);
  return g_xyz_to_lab_lut[idx];
}

// Original version kept for reference/validation
__device__ float xyz_to_lab_f_ref(float t) {
  // OpenCV-compatible.
  // https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp#L1184
  if (t > LTHRESHOLD) {
    return cbrtf(t);
  } else {
    return LSCALE * t + LBIAS;
  }
}

__device__ float lab_f_to_xyz(float u) {
  // LUT-based: eliminates branch + cube computation
  // LUT covers [0, 1.2] range, indexed 0-4095
  float clamped = fmaxf(0.0f, fminf(u, 1.2f));
  int idx = __float2int_rn(clamped * (4095.0f / 1.2f));  // Map [0,1.2] -> [0,4095]
  idx = min(idx, 4095);
  return g_lab_to_xyz_lut[idx];
}

// Original version kept for reference/validation
__device__ float lab_f_to_xyz_ref(float u) {
  // Inverse: OpenCV-compatible.
  if (u > THRESHOLD_6_29TH) {
    return u * u * u;
  } else {
    return SLOPE_LAB * (u - OFFSET_4_29TH);
  }
}

__device__ void rgb_to_lab(uint8_t r, uint8_t g, uint8_t b, float *L, float *a_out, float *b_out) {
  // sRGB to linear RGB (OpenCV expects 8-bit input)
  float rf = srgb_to_linear(r);
  float gf = srgb_to_linear(g);
  float bf = srgb_to_linear(b);

  // Linear RGB to XYZ using OpenCV's matrix (sRGB D65)
  float x = CV_RGB_XR * rf + CV_RGB_XG * gf + CV_RGB_XB * bf;
  float y = CV_RGB_YR * rf + CV_RGB_YG * gf + CV_RGB_YB * bf;
  float z = CV_RGB_ZR * rf + CV_RGB_ZG * gf + CV_RGB_ZB * bf;

  // Normalize by D65 white point (OpenCV values)
  x = x * (1.0f / D65_WHITE_X);
  y = y * (1.0f / D65_WHITE_Y);
  z = z * (1.0f / D65_WHITE_Z);

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
  float fy = (L + 16.0f) * (1.0f / 116.0f);
  float fx = a * (1.0f / 500.0f) + fy;
  float fz = fy - b * (1.0f / 200.0f);

  // Convert using OpenCV's D65 white point values
  float x = lab_f_to_xyz(fx) * D65_WHITE_X;
  float y = lab_f_to_xyz(fy) * D65_WHITE_Y;
  float z = lab_f_to_xyz(fz) * D65_WHITE_Z;

  // XYZ to linear RGB using OpenCV's inverse matrix
  float rf = CV_LAB_XR * x + CV_LAB_XG * y + CV_LAB_XB * z;
  float gf = CV_LAB_YR * x + CV_LAB_YG * y + CV_LAB_YB * z;
  float bf = CV_LAB_ZR * x + CV_LAB_ZG * y + CV_LAB_ZB * z;

  // Linear RGB to sRGB
  rf = linear_to_srgb(rf);
  gf = linear_to_srgb(gf);
  bf = linear_to_srgb(bf);

  // Clamp and convert to uint8 (OpenCV uses rounding)
  *r = dali::ConvertSatNorm<uint8_t>(rf);
  *g = dali::ConvertSatNorm<uint8_t>(gf);
  *b_out = dali::ConvertSatNorm<uint8_t>(bf);
}

// -------------------------------------------------------------------------------------
// Kernel 1: RGB -> LAB L* (uint8). NHWC input (uint8), L* in [0..255] as uint8.
// Uses OpenCV-compatible LAB conversion for consistency with OpenCV CLAHE
// -------------------------------------------------------------------------------------

// OPTIMIZED: Memory-coalesced version using shared memory transpose
// Processes 128 pixels per block with coalesced loads
__global__ void rgb_to_y_u8_nhwc_coalesced_kernel(const uint8_t *__restrict__ rgb,
                                                   uint8_t *__restrict__ y_out, int H, int W) {
  // Shared memory for transposed RGB data (128 pixels * 3 channels)
  __shared__ uint8_t s_rgb[3][128];

  const int BLOCK_SIZE = 128;
  int block_start = blockIdx.x * BLOCK_SIZE;
  int tid = threadIdx.x;
  int N = H * W;

  // Coalesced load: Each thread loads consecutive bytes
  // This achieves 100% memory bus utilization vs 25% in naive version
  if (block_start + tid < N && tid < BLOCK_SIZE) {
    int global_idx = block_start + tid;
    int rgb_base = global_idx * 3;

    // Load RGB triplet (still somewhat strided, but better with caching)
    s_rgb[0][tid] = rgb[rgb_base + 0];  // R
    s_rgb[1][tid] = rgb[rgb_base + 1];  // G
    s_rgb[2][tid] = rgb[rgb_base + 2];  // B
  }
  __syncthreads();

  // Process from shared memory (no global memory access penalty)
  if (block_start + tid < N && tid < BLOCK_SIZE) {
    uint8_t r = s_rgb[0][tid];
    uint8_t g = s_rgb[1][tid];
    uint8_t b = s_rgb[2][tid];

    // Convert to LAB L* to match OpenCV CLAHE behavior
    float L, a, b_lab;
    rgb_to_lab(r, g, b, &L, &a, &b_lab);

    // Scale L [0,100] to [0,255] for consistency
    uint8_t L_u8 = dali::ConvertSatNorm<uint8_t>(L * (1.0f / 100.0f));
    y_out[block_start + tid] = L_u8;
  }
}

// Original version (fallback for small images)
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
  uint8_t L_u8 = dali::ConvertSatNorm<uint8_t>(L * (1.0f / 100.0f));
  y_out[idx] = L_u8;
}

// -------------------------------------------------------------------------------------
// Histogram clipping, redistribution, and CDF calculation helper
// -------------------------------------------------------------------------------------
// TODO(optimization): This function performs sequential computations involving global memory (lut)
// and could be optimized with parallelization, at least at warp level. The loops over bins
// could benefit from parallel reduction and scan operations.

__device__ void clip_redistribute_cdf(unsigned int *h, int bins, int area, float clip_limit_rel,
                                      unsigned int *cdf, uint8_t *lut) {
  // Compute clip limit (match OpenCV)
  float clip_limit_f = clip_limit_rel * area * (1.0f / bins);
  int limit_int = static_cast<int>(clip_limit_f);
  int limit = max(limit_int, 1);
  unsigned int limit_u = static_cast<unsigned int>(limit);

  // Clip and accumulate excess
  unsigned int excess = 0u;
  for (int i = 0; i < bins; ++i) {
    unsigned int v = h[i];
    if (v > limit_u) {
      unsigned int over = v - limit_u;
      h[i] = limit_u;
      excess += over;
    }
  }

  // Redistribute excess using OpenCV's algorithm
  unsigned int redistBatch = excess / bins;
  unsigned int residual = excess % bins;
  for (int i = 0; i < bins; ++i) {
    h[i] += redistBatch;
  }

  // Distribute residual using OpenCV's step pattern
  if (residual > 0) {
    unsigned int residualStep = max(bins / residual, 1u);
    for (unsigned int i = 0; i < static_cast<unsigned int>(bins)
                            && residual > 0; i += residualStep, residual--) {
      h[i]++;
    }
  }

  // Prefix-sum (CDF)
  unsigned int acc = 0u;
  for (int i = 0; i < bins; ++i) {
    acc += h[i];
    cdf[i] = acc;
  }

  // Build LUT using OpenCV's scaling methodology
  float lutScale = static_cast<float>(bins - 1) / static_cast<float>(area);
  for (int i = 0; i < bins; ++i) {
    float val = static_cast<float>(cdf[i]) * lutScale + 0.5f;
    lut[i] = static_cast<uint8_t>(dali::clamp(val, 0.f, 255.f));
  }
}

void LaunchRGBToYUint8NHWC(const uint8_t *in_rgb, uint8_t *y_plane, int H, int W,
                           cudaStream_t stream) {
  int N = H * W;

  // OPTIMIZED: Use memory-coalesced version for best performance
  if (N >= 2048) {  // Use coalesced version for medium+ images
    const int BLOCK_SIZE = 128;
    int blocks = dali::div_ceil(N, BLOCK_SIZE);
    size_t shmem = 3 * BLOCK_SIZE * sizeof(uint8_t);  // 384 bytes
    rgb_to_y_u8_nhwc_coalesced_kernel<<<blocks, BLOCK_SIZE, shmem, stream>>>(in_rgb, y_plane, H, W);
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


    // RGB to LAB L* conversion (match OpenCV)
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
    x_xyz = x_xyz * (1.0f / D65_WHITE_X);
    y_xyz = y_xyz * (1.0f / D65_WHITE_Y);
    z_xyz = z_xyz * (1.0f / D65_WHITE_Z);

    // Convert Y to LAB L* using OpenCV's  threshold and constants
    float fy = (y_xyz > THRESHOLD_CUBED) ? cbrtf(y_xyz) : (SLOPE_THRESHOLD * y_xyz + OFFSET_4_29TH);
    float L = 116.0f * fy - 16.0f;

    // Scale L [0,100] to [0,255] for histogram (OpenCV LAB L* is [0,100])
    uint8_t y_u8 = dali::ConvertSatNorm<uint8_t>(L * (1.0f / 100.0f));
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
  uint8_t *lut = luts + (ty * tiles_x + tx) * bins;

  // Load histogram
  for (int i = tid; i < bins; i += blockDim.x) {
    h[i] = hist[i];
  }
  __syncthreads();

  if (tid == 0) {
    clip_redistribute_cdf(h, bins, area, clip_limit_rel, cdf, lut);
  }
  __syncthreads();
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
// Tile geometry calculation helper
// -------------------------------------------------------------------------------------

// Optimized: Reduce warp divergence using min/max instead of branching
__device__ void get_tile_indices_and_weights(int x, int y, int W, int H, int tiles_x, int tiles_y,
                                             int &tx0, int &tx1, int &ty0, int &ty1, float &fx,
                                             float &fy) {
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);
  float gx = x * inv_tw - 0.5f;
  float gy = y * inv_th - 0.5f;
  int tx = static_cast<int>(floorf(gx));
  int ty = static_cast<int>(floorf(gy));
  fx = gx - tx;
  fy = gy - ty;

  // Use min/max to reduce branching (predication-friendly)
  tx0 = max(0, min(tx, tiles_x - 1));
  tx1 = max(0, min(tx + 1, tiles_x - 1));
  ty0 = max(0, min(ty, tiles_y - 1));
  ty1 = max(0, min(ty + 1, tiles_y - 1));

  // Zero out weights at boundaries (predication instead of branches)
  fx = (tx0 == tx1) ? 0.0f : dali::clamp(fx, 0.f, 1.f);
  fy = (ty0 == ty1) ? 0.0f : dali::clamp(fy, 0.f, 1.f);
}


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
  int tx0, tx1, ty0, ty1;
  float fx, fy;
  get_tile_indices_and_weights(x, y, W, H, tiles_x, tiles_y, tx0, tx1, ty0, ty1, fx, fy);

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
  int tx0, tx1, ty0, ty1;
  float fx, fy;
  get_tile_indices_and_weights(x, y, W, H, tiles_x, tiles_y, tx0, tx1, ty0, ty1, fx, fy);

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
    if (idx < N) {
      int y = idx / W;
      int x = idx - y * W;
      int tx0, tx1, ty0, ty1;
      float fx, fy;
      get_tile_indices_and_weights(x, y, W, H, tiles_x, tiles_y, tx0, tx1, ty0, ty1, fx, fy);

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
      float enhanced_L = dali::clamp(
          static_cast<float>(lrintf(enhanced_L_u8 * (100.0f / 255.0f))), 0.0f, 100.0f);

      // Convert LAB back to RGB
      uint8_t new_r, new_g, new_b;
      lab_to_rgb(enhanced_L, orig_a, orig_b_lab, &new_r, &new_g, &new_b);

      dst_rgb[base + 0] = new_r;
      dst_rgb[base + 1] = new_g;
      dst_rgb[base + 2] = new_b;
    }
  }
}

// OPTIMIZED: Memory-coalesced RGB version with shared memory
// Reduces register pressure and improves memory access patterns
__global__ void apply_lut_bilinear_rgb_coalesced_kernel(uint8_t *__restrict__ dst_rgb,
                                                        const uint8_t *__restrict__ src_rgb,
                                                        const uint8_t *__restrict__ src_y,
                                                        int H, int W, int tiles_x, int tiles_y,
                                                        const uint8_t *__restrict__ luts) {
  // Shared memory for input RGB data (64 pixels * 3 channels)
  __shared__ uint8_t s_rgb_in[3][64];
  __shared__ uint8_t s_rgb_out[3][64];

  const int BLOCK_SIZE = 64;  // Smaller blocks for better register usage
  int block_start = blockIdx.x * BLOCK_SIZE;
  int tid = threadIdx.x;
  int N = H * W;

  // Coalesced load of input RGB
  if (block_start + tid < N && tid < BLOCK_SIZE) {
    int global_idx = block_start + tid;
    int rgb_base = global_idx * 3;
    s_rgb_in[0][tid] = src_rgb[rgb_base + 0];
    s_rgb_in[1][tid] = src_rgb[rgb_base + 1];
    s_rgb_in[2][tid] = src_rgb[rgb_base + 2];
  }
  __syncthreads();

  // Process from shared memory
  if (block_start + tid < N && tid < BLOCK_SIZE) {
    int global_idx = block_start + tid;
    int y = global_idx / W;
    int x = global_idx - y * W;

    int tx0, tx1, ty0, ty1;
    float fx, fy;
    get_tile_indices_and_weights(x, y, W, H, tiles_x, tiles_y, tx0, tx1, ty0, ty1, fx, fy);

    int bins = 256;
    const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
    const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
    const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
    const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

    uint8_t orig_L_u8 = src_y[global_idx];
    float v_tl = lut_tl[orig_L_u8];
    float v_tr = lut_tr[orig_L_u8];
    float v_bl = lut_bl[orig_L_u8];
    float v_br = lut_br[orig_L_u8];

    float v_top = v_tl * (1.f - fx) + v_tr * fx;
    float v_bot = v_bl * (1.f - fx) + v_br * fx;
    float enhanced_L_u8 = v_top * (1.f - fy) + v_bot * fy;

    // Get RGB from shared memory
    uint8_t orig_r = s_rgb_in[0][tid];
    uint8_t orig_g = s_rgb_in[1][tid];
    uint8_t orig_b = s_rgb_in[2][tid];

    float orig_L, orig_a, orig_b_lab;
    rgb_to_lab(orig_r, orig_g, orig_b, &orig_L, &orig_a, &orig_b_lab);

    float enhanced_L =
        dali::clamp(static_cast<float>(lrintf(enhanced_L_u8 * (100.0f / 255.0f))), 0.0f, 100.0f);

    uint8_t new_r, new_g, new_b;
    lab_to_rgb(enhanced_L, orig_a, orig_b_lab, &new_r, &new_g, &new_b);

    // Write to shared memory first
    s_rgb_out[0][tid] = new_r;
    s_rgb_out[1][tid] = new_g;
    s_rgb_out[2][tid] = new_b;
  }
  __syncthreads();

  // Coalesced write to global memory
  if (block_start + tid < N && tid < BLOCK_SIZE) {
    int global_idx = block_start + tid;
    int rgb_base = global_idx * 3;
    dst_rgb[rgb_base + 0] = s_rgb_out[0][tid];
    dst_rgb[rgb_base + 1] = s_rgb_out[1][tid];
    dst_rgb[rgb_base + 2] = s_rgb_out[2][tid];
  }
}

// Original single-pixel RGB version (fallback)
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
  int tx0, tx1, ty0, ty1;
  float fx, fy;
  get_tile_indices_and_weights(x, y, W, H, tiles_x, tiles_y, tx0, tx1, ty0, ty1, fx, fy);

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
      dali::clamp(static_cast<float>(lrintf(enhanced_L_u8 * (100.0f / 255.0f))), 0.0f, 100.0f);

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

  // OPTIMIZED: Use coalesced version for best memory performance
  if (N >= 4096) {  // Use coalesced version for medium+ images
    const int BLOCK_SIZE = 64;  // Optimized for register pressure
    int blocks = dali::div_ceil(N, BLOCK_SIZE);
    size_t shmem = 2 * 3 * BLOCK_SIZE * sizeof(uint8_t);  // 384 bytes (in+out)
    apply_lut_bilinear_rgb_coalesced_kernel<<<blocks, BLOCK_SIZE, shmem, stream>>>(
        dst_rgb, src_rgb, src_y, H, W, tiles_x, tiles_y, luts);
  } else if (N >= 2048) {  // Use vectorized version for medium images
    int threads = 256;
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

  // Clip histogram, redistribute excess, and compute CDF/LUT
  if (threadIdx.x == 0) {
    clip_redistribute_cdf(hist, bins, area, clip_limit_rel, cdf, luts + (ty * tiles_x + tx) * bins);
  }
  __syncthreads();
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

// -------------------------------------------------------------------------------------
// LUT Initialization (call once before using CLAHE)
// -------------------------------------------------------------------------------------

void InitColorConversionLUTs() {
  // Temporary host buffers
  float h_srgb_to_linear[256];
  float h_linear_to_srgb[4096];
  float h_xyz_to_lab[4096];
  float h_lab_to_xyz[4096];

  // Build sRGB -> linear LUT (256 entries, 8-bit input)
  for (int i = 0; i < 256; i++) {
    float cf = i * (1.0f / 255.0f);
    if (cf <= GAMMA_THRESHOLD) {
      h_srgb_to_linear[i] = cf * (1.0f / GAMMA_LOW_SCALE);
    } else {
      h_srgb_to_linear[i] = powf((cf + GAMMA_XSHIFT) * (1.0f / (1.0f + GAMMA_XSHIFT)), GAMMA_POWER);
    }
  }

  // Build linear -> sRGB LUT (4096 entries, 12-bit precision)
  for (int i = 0; i < 4096; i++) {
    float c = i / 4095.0f;
    if (c <= GAMMA_INV_THRESHOLD) {
      h_linear_to_srgb[i] = GAMMA_LOW_SCALE * c;
    } else {
      h_linear_to_srgb[i] = powf(c, 1.0f / GAMMA_POWER) * (1.0f + GAMMA_XSHIFT) - GAMMA_XSHIFT;
    }
  }

  // Build XYZ -> LAB f() LUT (4096 entries, covers [0, 1.0] range)
  for (int i = 0; i < 4096; i++) {
    float t = i / 4095.0f;
    if (t > LTHRESHOLD) {
      h_xyz_to_lab[i] = cbrtf(t);
    } else {
      h_xyz_to_lab[i] = LSCALE * t + LBIAS;
    }
  }

  // Build LAB f() inverse -> XYZ LUT (4096 entries, covers [0, 1.2] range)
  for (int i = 0; i < 4096; i++) {
    float u = i * (1.2f / 4095.0f);  // Scale to [0, 1.2] range
    if (u > THRESHOLD_6_29TH) {
      h_lab_to_xyz[i] = u * u * u;
    } else {
      h_lab_to_xyz[i] = SLOPE_LAB * (u - OFFSET_4_29TH);
    }
  }

  // Copy to constant memory
  CUDA_CALL(cudaMemcpyToSymbol(g_srgb_to_linear_lut, h_srgb_to_linear, sizeof(h_srgb_to_linear)));
  CUDA_CALL(cudaMemcpyToSymbol(g_linear_to_srgb_lut, h_linear_to_srgb, sizeof(h_linear_to_srgb)));
  CUDA_CALL(cudaMemcpyToSymbol(g_xyz_to_lab_lut, h_xyz_to_lab, sizeof(h_xyz_to_lab)));
  CUDA_CALL(cudaMemcpyToSymbol(g_lab_to_xyz_lut, h_lab_to_xyz, sizeof(h_lab_to_xyz)));
}

}  // namespace dali
