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

#define THRESHOLD_6_29TH (6.0f / 29.0f)
#define OFFSET_4_29TH (4.0f / 29.0f)
#define SLOPE_841_108TH (841.0f / 108.0f)  // (29/6)^2 / 3

// -------------------------------------------------------------------------------------
// Helper functions for RGB ↔ LAB conversion (match OpenCV exactly)
// -------------------------------------------------------------------------------------
__device__ float srgb_to_linear(float c) {
  // OpenCV's exact gamma correction
  return (c > 0.04045f) ? powf((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
}

__device__ float linear_to_srgb(float c) {
  // OpenCV's exact inverse gamma correction
  return (c > 0.0031308f) ? 1.055f * powf(c, 1.0f / 2.4f) - 0.055f : 12.92f * c;
}

__device__ float xyz_to_lab_f(float t) {
  // δ = 6/29;  compare t to δ^3; slope = (1/3)*(29/6)^2.  OpenCV-compatible.
  const float delta = THRESHOLD_6_29TH;
  const float threshold = delta * delta * delta;           // δ^3
  const float slope = (1.0f / 3.0f) * (1.0f / (delta * delta));  // (29/6)^2 / 3
  return (t > threshold) ? cbrtf(t) : (slope * t + OFFSET_4_29TH);
}

__device__ float lab_f_to_xyz(float u) {
  // Inverse: compare u to δ; slope = 3*δ^2.  OpenCV-compatible.
  const float delta = THRESHOLD_6_29TH;
  const float threshold = delta;                            // compare f(Y) to δ
  const float slope = 3.0f * delta * delta;                 // 3*δ^2
  return (u > threshold) ? (u * u * u) : (slope * (u - OFFSET_4_29TH));
}

__device__ void rgb_to_lab(uint8_t r, uint8_t g, uint8_t b,
                           float *L, float *a_out, float *b_out) {
  // Normalize to [0,1]
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  // sRGB to linear RGB
  rf = srgb_to_linear(rf);
  gf = srgb_to_linear(gf);
  bf = srgb_to_linear(bf);

  // Linear RGB to XYZ using OpenCV's exact matrix (sRGB D65)
  float x = 0.4124564390896922f * rf
          + 0.3575761206819519f * gf
          + 0.1804375005091677f * bf;
  float y = 0.2126728514056224f * rf
          + 0.7151579067501442f * gf
          + 0.0721690406852293f * bf;
  float z = 0.0193338958834121f * rf
          + 0.1191920336965374f * gf
          + 0.9503040785363140f * bf;

  // Normalize by D65 white point (OpenCV exact values)
  x = x / 0.9504559270516716f;
  y = y / 1.0000000000000000f;
  z = z / 1.0890577507598784f;

  // XYZ to LAB
  float fx = xyz_to_lab_f(x);
  float fy = xyz_to_lab_f(y);
  float fz = xyz_to_lab_f(z);

  *L = 116.0f * fy - 16.0f;
  *a_out = 500.0f * (fx - fy);
  *b_out = 200.0f * (fy - fz);
}

__device__ void lab_to_rgb(float L, float a, float b,
                           uint8_t *r, uint8_t *g, uint8_t *b_out) {
  // LAB to XYZ
  float fy = (L + 16.0f) / 116.0f;
  float fx = a / 500.0f + fy;
  float fz = fy - b / 200.0f;

  // Convert using OpenCV's exact D65 white point values
  float x = lab_f_to_xyz(fx) * 0.9504559270516716f;
  float y = lab_f_to_xyz(fy) * 1.0000000000000000f;
  float z = lab_f_to_xyz(fz) * 1.0890577507598784f;

  // XYZ to linear RGB using OpenCV's exact inverse matrix
  float rf = 3.2404541621141045f * x
           - 1.5371385127977166f * y
           - 0.4985314095560162f * z;
  float gf = -0.9692660305051868f * x
           + 1.8760108454466942f * y
           + 0.0415560175303051f * z;
  float bf = 0.0556434309971394f * x
           - 0.2040259135167538f * y
           + 1.0572251882231791f * z;

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
  if (idx >= N) { return; }

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
    if (idx >= N) { return; }

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
  if (tx >= tiles_x || ty >= tiles_y) { return; }

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x) { shist[i] = 0u; }
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

    // RGB to LAB L* conversion (match OpenCV exactly)
    // First convert to normalized RGB [0,1]
    float r_val = rgb[rgb_idx + 0];
    float g_val = rgb[rgb_idx + 1];
    float b_val = rgb[rgb_idx + 2];

    float rf = r_val / 255.0f;
    float gf = g_val / 255.0f;
    float bf = b_val / 255.0f;

    // Apply gamma correction (sRGB to linear RGB)
    rf = srgb_to_linear(rf);
    gf = srgb_to_linear(gf);
    bf = srgb_to_linear(bf);

    // Convert to CIE XYZ using OpenCV's exact transformation matrix
    float x_xyz = 0.4124564390896922f * rf
                + 0.3575761206819519f * gf
                + 0.1804375005091677f * bf;
    float y_xyz = 0.2126728514056224f * rf
                + 0.7151579067501442f * gf
                + 0.0721690406852293f * bf;
    float z_xyz = 0.0193338958834121f * rf
                + 0.1191920336965374f * gf
                + 0.9503040785363140f * bf;

    // Normalize by D65 white point (OpenCV exact values)
    x_xyz = x_xyz / 0.9504559270516716f;
    y_xyz = y_xyz / 1.0000000000000000f;
    z_xyz = z_xyz / 1.0890577507598784f;

    // Convert Y to LAB L* using OpenCV's exact threshold and constants
    const float threshold = THRESHOLD_6_29TH * THRESHOLD_6_29TH * THRESHOLD_6_29TH;  // δ^3
    float fy = (y_xyz > threshold) ? cbrtf(y_xyz) : (SLOPE_841_108TH * y_xyz + OFFSET_4_29TH);
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
  for (int i = threadIdx.x; i < bins; i += blockDim.x) { g_hist[i] = shist[i]; }
}

void LaunchFusedRGBToYHist(const uint8_t *rgb, uint8_t *y_plane, int H, int W, int tiles_x,
                           int tiles_y, unsigned int *histograms, cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // Optimized for both compute and shared memory
  size_t shmem = 256 * sizeof(unsigned int);
  fused_rgb_to_y_hist_kernel<<<grid, threads, shmem, stream>>>(
      rgb, y_plane, H, W, tiles_x, tiles_y, histograms);
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
  if (tx >= tiles_x || ty >= tiles_y) { return; }

  int warp_id = threadIdx.x / warp_size;
  int lane_id = threadIdx.x % warp_size;

  // Per-warp private histograms (warps_per_block * 256 bins)
  // This reduces atomic contention significantly
  unsigned int *warp_hist = shist + warp_id * bins;
  unsigned int *global_hist = shist + warps_per_block * bins;  // Final merged histogram

  // Zero per-warp histogram
  for (int i = lane_id; i < bins; i += warp_size) { warp_hist[i] = 0u; }

  // Zero global histogram (only first warp)
  if (warp_id == 0) {
    for (int i = lane_id; i < bins; i += warp_size) { global_hist[i] = 0u; }
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
    for (int w = 0; w < warps_per_block; ++w) { sum += shist[w * bins + bin]; }
    global_hist[bin] = sum;
  }
  __syncthreads();

  // Write back to global memory
  unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
  for (int i = threadIdx.x; i < bins; i += blockDim.x) { g_hist[i] = global_hist[i]; }
}

void LaunchHistPerTile256WarpOptimized(const uint8_t *y_plane, int H, int W, int tiles_x,
                                       int tiles_y, unsigned int *histograms, cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // 16 warps per block
  int warps_per_block = threads / 32;
  // Shared memory: warps_per_block * 256 (private) + 256 (final)
  size_t shmem = (warps_per_block + 1) * 256 * sizeof(unsigned int);
  hist_per_tile_256_warp_optimized_kernel<<<grid, threads, shmem, stream>>>(
      y_plane, H, W, tiles_x, tiles_y, histograms);
}

// Original version kept for fallback
__global__ void hist_per_tile_256_kernel(const uint8_t *__restrict__ y_plane, int H, int W,
                                         int tiles_x, int tiles_y,
                                         unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y) { return; }

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x) { shist[i] = 0u; }
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
  for (int i = threadIdx.x; i < bins; i += blockDim.x) { g_hist[i] = shist[i]; }
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
    hist_per_tile_256_kernel<<<grid, threads, shmem, stream>>>(
        y_plane, H, W, tiles_x, tiles_y, histograms);
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
  if (tx >= tiles_x || ty >= tiles_y) { return; }

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
  for (int i = tid; i < bins; i += blockDim.x) { h[i] = hist[i]; }
  __syncthreads();

  // Compute clip limit (match OpenCV exactly)
  float clip_limit_f = clip_limit_rel * area / bins;
  int limit_int = static_cast<int>(clip_limit_f);
  int limit = max(limit_int, 1);
  unsigned int limit_u = static_cast<unsigned int>(limit);

  // Clip and accumulate excess
  __shared__ unsigned int excess;
  if (tid == 0) { excess = 0u; }
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

  // Redistribute excess using OpenCV's exact algorithm
  unsigned int redistBatch = excess / bins;   // OpenCV: redistBatch = clipped / histSize
  unsigned int residual = excess % bins;      // OpenCV: residual = clipped - redistBatch * histSize

  for (int i = tid; i < bins; i += blockDim.x) { h[i] += redistBatch; }
  __syncthreads();

  // Distribute residual using OpenCV's exact step pattern
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

  // Build LUT using OpenCV's exact scaling methodology
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
  clip_cdf_lut_256_kernel<<<grid, threads, 0, stream>>>(
      histograms, tiles_x, tiles_y, tile_w, tile_h, W, H, clip_limit_rel, luts);
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
    if (idx >= N) { return; }

    int y = idx / W;
    int x = idx - y * W;

    // Tile geometry - match OpenCV exactly (same as RGB version)
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
  if (idx >= N) { return; }

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - match OpenCV exactly (same as RGB version)
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);  // 1.0f / tileSize.width
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);  // 1.0f / tileSize.height

  // Tile coordinates (match OpenCV exactly)
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
  if (idx >= N) { return; }

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - match OpenCV exactly
  float inv_tw = static_cast<float>(tiles_x) / static_cast<float>(W);
  float inv_th = static_cast<float>(tiles_y) / static_cast<float>(H);

  // Tile coordinates (match OpenCV exactly)
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
    apply_lut_bilinear_gray_kernel<<<blocks, threads, 0, stream>>>(
        dst_gray, src_gray, H, W, tiles_x, tiles_y, luts);
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
    if (idx >= N) { return; }

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
    float enhanced_L = dali::clamp(
        static_cast<float>(lrintf(enhanced_L_u8 * 100.0f / 255.0f)),
        0.0f, 100.0f);

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
  if (idx >= N) { return; }

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
  float enhanced_L = dali::clamp(
      static_cast<float>(lrintf(enhanced_L_u8 * 100.0f / 255.0f)),
      0.0f, 100.0f);

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
  if (N >= 8192) {                               // Threshold for vectorized processing
    int threads = 256;                           // Better occupancy with complex RGB processing
    int blocks = dali::div_ceil(N, threads * 2);  // Each thread processes 2 pixels
    apply_lut_bilinear_rgb_vectorized_kernel<<<blocks, threads, 0, stream>>>(
        dst_rgb, src_rgb, src_y, H, W, tiles_x, tiles_y, luts);
  } else {
    // Use original version for smaller images
    int threads = 512;
    int blocks = dali::div_ceil(N, threads);
    apply_lut_bilinear_rgb_kernel<<<blocks, threads, 0, stream>>>(
        dst_rgb, src_rgb, src_y, H, W, tiles_x, tiles_y, luts);
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
  if (tx >= tiles_x || ty >= tiles_y) { return; }

  int warp_id = threadIdx.x / warp_size;
  int lane_id = threadIdx.x % warp_size;

  // Initialize shared memory
  unsigned int *my_warp_hist = warp_hist + warp_id * bins;
  for (int i = lane_id; i < bins; i += warp_size) { my_warp_hist[i] = 0u; }

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
  for (int w = 0; w < warps_per_block; ++w) { sum += warp_hist[w * bins + bin]; }
    hist[bin] = sum;
  }
  __syncthreads();

  // Clip histogram and redistribute excess
  float clip_limit_f = clip_limit_rel * area / bins;
  unsigned int limit = max(static_cast<unsigned int>(clip_limit_f), 1u);

  __shared__ unsigned int excess;
  if (threadIdx.x == 0) { excess = 0u; }
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

  for (int i = threadIdx.x; i < bins; i += blockDim.x) { hist[i] += redistBatch; }
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
    LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts,
                          stream);
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
