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

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                                  \
  do {                                                                                    \
    cudaError_t __err = (expr);                                                           \
    if (__err != cudaSuccess) {                                                           \
      printf("CUDA error %d at %s:%d: %s\n", static_cast<int>(__err), __FILE__, __LINE__, \
             cudaGetErrorString(__err));                                                  \
    }                                                                                     \
  } while (0)
#endif

static inline __host__ __device__ int div_up(int a, int b) {
  return (a + b - 1) / b;
}

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
  // OpenCV uses these exact thresholds and constants
  const float delta = 6.0f / 29.0f;
  const float delta_cube = delta * delta * delta;
  return (t > delta_cube) ? cbrtf(t) : (t / (3.0f * delta * delta) + 4.0f / 29.0f);
}

__device__ float lab_f_to_xyz(float t) {
  // OpenCV's exact inverse transformation
  const float delta = 6.0f / 29.0f;
  return (t > delta) ? (t * t * t) : (3.0f * delta * delta * (t - 4.0f / 29.0f));
}

__device__ void rgb_to_lab(uint8_t r, uint8_t g, uint8_t b, float *L, float *a_out, float *b_out) {
  // Normalize to [0,1]
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;

  // sRGB to linear RGB
  rf = srgb_to_linear(rf);
  gf = srgb_to_linear(gf);
  bf = srgb_to_linear(bf);

  // Linear RGB to XYZ using OpenCV's exact matrix (sRGB D65)
  float x = 0.412453f * rf + 0.357580f * gf + 0.180423f * bf;
  float y = 0.212671f * rf + 0.715160f * gf + 0.072169f * bf;
  float z = 0.019334f * rf + 0.119193f * gf + 0.950227f * bf;

  // Normalize by D65 white point (OpenCV values)
  x = x / 0.950456f;
  y = y / 1.000000f;
  z = z / 1.088754f;

  // XYZ to LAB
  float fx = xyz_to_lab_f(x);
  float fy = xyz_to_lab_f(y);
  float fz = xyz_to_lab_f(z);

  *L = 116.0f * fy - 16.0f;
  *a_out = 500.0f * (fx - fy);
  *b_out = 200.0f * (fy - fz);
}

__device__ void lab_to_rgb(float L, float a, float b, uint8_t *r, uint8_t *g, uint8_t *b_out) {
  // LAB to XYZ
  float fy = (L + 16.0f) / 116.0f;
  float fx = a / 500.0f + fy;
  float fz = fy - b / 200.0f;

  // Convert using OpenCV's D65 white point
  float x = lab_f_to_xyz(fx) * 0.950456f;
  float y = lab_f_to_xyz(fy) * 1.000000f;
  float z = lab_f_to_xyz(fz) * 1.088754f;

  // XYZ to linear RGB using OpenCV's exact inverse matrix
  float rf = 3.240479f * x - 1.537150f * y - 0.498535f * z;
  float gf = -0.969256f * x + 1.875991f * y + 0.041556f * z;
  float bf = 0.055648f * x - 0.204043f * y + 1.057311f * z;

  // Linear RGB to sRGB
  rf = linear_to_srgb(rf);
  gf = linear_to_srgb(gf);
  bf = linear_to_srgb(bf);

  // Clamp and convert to uint8
  *r = (uint8_t)lrintf(fminf(fmaxf(rf * 255.0f, 0.f), 255.f));
  *g = (uint8_t)lrintf(fminf(fmaxf(gf * 255.0f, 0.f), 255.f));
  *b_out = (uint8_t)lrintf(fminf(fmaxf(bf * 255.0f, 0.f), 255.f));
}

// -------------------------------------------------------------------------------------
// Kernel 1: RGB -> Y (uint8). NHWC input (uint8), Y in [0..255] as uint8.
// BT.601 luma: Y = 0.299 R + 0.587 G + 0.114 B
// -------------------------------------------------------------------------------------
__global__ void rgb_to_y_u8_nhwc_kernel(const uint8_t *__restrict__ rgb,
                                        uint8_t *__restrict__ y_out, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N)
    return;

  int c0 = 3 * idx;
  float r = rgb[c0 + 0];
  float g = rgb[c0 + 1];
  float b = rgb[c0 + 2];

  float y = 0.299f * r + 0.587f * g + 0.114f * b;
  int yi = static_cast<int>(lrintf(fminf(fmaxf(y, 0.f), 255.f)));
  y_out[idx] = static_cast<uint8_t>(yi);
}

// Vectorized version for better memory bandwidth (processes 4 pixels at once)
__global__ void rgb_to_y_u8_nhwc_vectorized_kernel(const uint8_t *__restrict__ rgb,
                                                   uint8_t *__restrict__ y_out, int H, int W) {
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int N = H * W;

  // Process 4 pixels per thread for better memory coalescing
  for (int i = 0; i < 4 && (base_idx + i) < N; ++i) {
    int idx = base_idx + i;
    int c0 = 3 * idx;

    float r = rgb[c0 + 0];
    float g = rgb[c0 + 1];
    float b = rgb[c0 + 2];

    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    int yi = static_cast<int>(lrintf(fminf(fmaxf(y, 0.f), 255.f)));
    y_out[idx] = static_cast<uint8_t>(yi);
  }
}

extern "C" void LaunchRGBToYUint8NHWC(const uint8_t *in_rgb, uint8_t *y_plane, int H, int W,
                                      cudaStream_t stream) {
  int N = H * W;

  // Use vectorized version for better memory bandwidth on larger images
  if (N >= 4096) {      // Threshold for using vectorized version
    int threads = 256;  // Each thread processes 4 pixels
    int blocks = div_up(N, threads * 4);
    rgb_to_y_u8_nhwc_vectorized_kernel<<<blocks, threads, 0, stream>>>(in_rgb, y_plane, H, W);
  } else {
    int threads = 512;  // Standard version
    int blocks = div_up(N, threads);
    rgb_to_y_u8_nhwc_kernel<<<blocks, threads, 0, stream>>>(in_rgb, y_plane, H, W);
  }
}

// -------------------------------------------------------------------------------------
// Fused Kernel: RGB to Y + Histogram per tile (optimized)
// Combines RGB->Y conversion with histogram computation to reduce memory round-trips
// Each block handles one tile and builds histogram from RGB data directly
// -------------------------------------------------------------------------------------
__global__ void fused_rgb_to_y_hist_kernel(const uint8_t *__restrict__ rgb,
                                           uint8_t *__restrict__ y_out, int H, int W, int tiles_x,
                                           int tiles_y, unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y)
    return;

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x)
    shist[i] = 0u;
  __syncthreads();

  // Compute tile bounds
  int tile_w = div_up(W, tiles_x);
  int tile_h = div_up(H, tiles_y);
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
    // From OpenCV source: cv::COLOR_RGB2Lab
    float x_xyz = 0.412453f * rf + 0.357580f * gf + 0.180423f * bf;
    float y_xyz = 0.212671f * rf + 0.715160f * gf + 0.072169f * bf;
    float z_xyz = 0.019334f * rf + 0.119193f * gf + 0.950227f * bf;

    // Normalize by D65 white point (OpenCV values)
    x_xyz = x_xyz / 0.950456f;
    y_xyz = y_xyz / 1.000000f;
    z_xyz = z_xyz / 1.088754f;

    // Convert Y to LAB L* using OpenCV's threshold and constants
    float fy = (y_xyz > 0.008856f) ? cbrtf(y_xyz) : (7.787f * y_xyz + 16.0f / 116.0f);
    float L = 116.0f * fy - 16.0f;

    // Scale L [0,100] to [0,255] for histogram (OpenCV LAB L* is [0,100])
    uint8_t y_u8 = (uint8_t)lrintf(fminf(fmaxf(L * 255.0f / 100.0f, 0.f), 255.f));  // Store Y value
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

extern "C" void LaunchFusedRGBToYHist(const uint8_t *rgb, uint8_t *y_plane, int H, int W,
                                      int tiles_x, int tiles_y, unsigned int *histograms,
                                      cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // Optimized for both compute and shared memory
  size_t shmem = 256 * sizeof(unsigned int);
  fused_rgb_to_y_hist_kernel<<<grid, threads, shmem, stream>>>(rgb, y_plane, H, W, tiles_x, tiles_y,
                                                               histograms);
}

// -------------------------------------------------------------------------------------
// Kernel 2: Histograms per tile (256 bins, uint32).
// One thread block per tile; shared 256-bin histogram.
// Each block sweeps its tile with a grid-stride loop.
// -------------------------------------------------------------------------------------
__global__ void hist_per_tile_256_kernel(const uint8_t *__restrict__ y_plane, int H, int W,
                                         int tiles_x, int tiles_y,
                                         unsigned int *__restrict__ histograms) {
  extern __shared__ unsigned int shist[];  // 256 bins
  const int bins = 256;

  int tx = blockIdx.x;  // tile x
  int ty = blockIdx.y;  // tile y
  if (tx >= tiles_x || ty >= tiles_y)
    return;

  // Zero shared histogram
  for (int i = threadIdx.x; i < bins; i += blockDim.x)
    shist[i] = 0u;
  __syncthreads();

  // Compute tile bounds
  int tile_w = div_up(W, tiles_x);
  int tile_h = div_up(H, tiles_y);
  int x0 = tx * tile_w;
  int y0 = ty * tile_h;
  int x1 = min(x0 + tile_w, W);
  int y1 = min(y0 + tile_h, H);

  // Loop over tile pixels
  int area = (x1 - x0) * (y1 - y0);
  for (int i = threadIdx.x; i < area; i += blockDim.x) {
    int dy = i / (x1 - x0);
    int dx = i - dy * (x1 - x0);
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

extern "C" void LaunchHistPerTile256(const uint8_t *y_plane, int H, int W, int tiles_x, int tiles_y,
                                     unsigned int *histograms, cudaStream_t stream) {
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // Increased for better shared memory utilization
  size_t shmem = 256 * sizeof(unsigned int);
  hist_per_tile_256_kernel<<<grid, threads, shmem, stream>>>(y_plane, H, W, tiles_x, tiles_y,
                                                             histograms);
}

// -------------------------------------------------------------------------------------
// Kernel 3: Clip + CDF -> LUT per tile (uint8 LUT).
// clip_limit_rel: relative multiplier of the average bin count per tile
//   limit = clip_limit_rel * (tile_area / bins)
// Excess is redistributed uniformly.
// LUT[v] = round( (cdf[v] - cdf_min) / (tile_area - cdf_min) * 255 )
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
  if (tx >= tiles_x || ty >= tiles_y)
    return;

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
  for (int i = tid; i < bins; i += blockDim.x)
    h[i] = hist[i];
  __syncthreads();

  // Compute clip limit (match OpenCV exactly)
  float clip_limit_f =
      clip_limit_rel * area / bins;  // OpenCV: clipLimit * tileSizeTotal / histSize
  unsigned int limit = static_cast<unsigned int>(clip_limit_f);
  limit = max(limit, 1u);  // OpenCV: std::max(clipLimit, 1)

  // Clip and accumulate excess
  __shared__ unsigned int excess;
  if (tid == 0)
    excess = 0u;
  __syncthreads();

  for (int i = tid; i < bins; i += blockDim.x) {
    unsigned int v = h[i];
    if (v > limit) {
      unsigned int over = v - limit;
      h[i] = limit;
      atomicAdd(&excess, over);
    }
  }
  __syncthreads();

  // Redistribute excess using OpenCV's exact algorithm
  unsigned int redistBatch = excess / bins;  // OpenCV: redistBatch = clipped / histSize
  unsigned int residual = excess % bins;     // OpenCV: residual = clipped - redistBatch * histSize

  for (int i = tid; i < bins; i += blockDim.x) {
    h[i] += redistBatch;  // OpenCV: tileHist[i] += redistBatch
  }
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

  // Build LUT using OpenCV's exact scaling
  uint8_t *lut = luts + (ty * tiles_x + tx) * bins;
  float lutScale = static_cast<float>(bins - 1) /
                   static_cast<float>(area);  // OpenCV: (histSize - 1) / tileSizeTotal

  for (int i = tid; i < bins; i += blockDim.x) {
    float val = static_cast<float>(cdf[i]) * lutScale;  // OpenCV: sum * lutScale
    lut[i] = static_cast<uint8_t>(lrintf(fminf(fmaxf(val, 0.f), 255.f)));
  }
}

extern "C" void LaunchClipCdfToLut256(unsigned int *histograms, int H, int W, int tiles_x,
                                      int tiles_y, float clip_limit_rel, uint8_t *luts,
                                      cudaStream_t stream) {
  int tile_w = div_up(W, tiles_x);
  int tile_h = div_up(H, tiles_y);
  dim3 grid(tiles_x, tiles_y, 1);
  int threads = 512;  // Increased for better compute utilization
  clip_cdf_lut_256_kernel<<<grid, threads, 0, stream>>>(histograms, tiles_x, tiles_y, tile_w,
                                                        tile_h, W, H, clip_limit_rel, luts);
}

// -------------------------------------------------------------------------------------
// Kernel 4a: Apply LUT with bilinear interpolation for GRAYSCALE output.
// For each pixel, look up 4 neighboring tile LUTs and bilinearly blend.
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_gray_kernel(const uint8_t *__restrict__ src_y,
                                               uint8_t *__restrict__ dst_y, int H, int W,
                                               int tiles_x, int tiles_y,
                                               const uint8_t *__restrict__ luts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N)
    return;

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - use same calculation as histogram kernel for consistency
  int tile_w = div_up(W, tiles_x);
  int tile_h = div_up(H, tiles_y);

  // Tile coordinates
  float gx = (x + 0.5f) / tile_w - 0.5f;  // tile-space x
  float gy = (y + 0.5f) / tile_h - 0.5f;  // tile-space y
  int tx = static_cast<int>(floorf(gx));
  int ty = static_cast<int>(floorf(gy));
  float fx = gx - tx;
  float fy = gy - ty;

  // Handle border cases properly
  // For pixels outside tile boundaries, use border extrapolation
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
    fx = fminf(fmaxf(fx, 0.f), 1.f);
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
    fy = fminf(fmaxf(fy, 0.f), 1.f);
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

  int outi = static_cast<int>(lrintf(fminf(fmaxf(v_out, 0.f), 255.f)));
  dst_y[idx] = (uint8_t)outi;
}

extern "C" void LaunchApplyLUTBilinearToGray(const uint8_t *src_gray, uint8_t *dst_gray, int H,
                                             int W, int tiles_x, int tiles_y, const uint8_t *luts,
                                             cudaStream_t stream) {
  int N = H * W;
  int threads = 512;  // Increased for better memory bandwidth utilization
  int blocks = div_up(N, threads);
  apply_lut_bilinear_gray_kernel<<<blocks, threads, 0, stream>>>(src_gray, dst_gray, H, W, tiles_x,
                                                                 tiles_y, luts);
}

// -------------------------------------------------------------------------------------
// Kernel 4b: Apply LUT for RGB using proper LAB color space (match OpenCV exactly)
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_rgb_kernel(const uint8_t *__restrict__ src_rgb,
                                              const uint8_t *__restrict__ src_y,  // original L*
                                              uint8_t *__restrict__ dst_rgb, int H, int W,
                                              int tiles_x, int tiles_y,
                                              const uint8_t *__restrict__ luts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = H * W;
  if (idx >= N)
    return;

  int y = idx / W;
  int x = idx - y * W;

  // Tile geometry - match OpenCV exactly
  // OpenCV: tileSize = Size(src.width / tilesX, src.height / tilesY)
  float inv_tw =
      static_cast<float>(tiles_x) / static_cast<float>(W);  // OpenCV: 1.0f / tileSize.width
  float inv_th =
      static_cast<float>(tiles_y) / static_cast<float>(H);  // OpenCV: 1.0f / tileSize.height

  // Tile coordinates (match OpenCV exactly)
  float txf = x * inv_tw - 0.5f;  // OpenCV: x * inv_tw - 0.5f
  float tyf = y * inv_th - 0.5f;  // OpenCV: y * inv_th - 0.5f

  int tx = static_cast<int>(floorf(txf));  // OpenCV: cvFloor(txf)
  int ty = static_cast<int>(floorf(tyf));  // OpenCV: cvFloor(tyf)
  float fx = txf - tx;                     // OpenCV: xa = txf - tx1
  float fy = tyf - ty;                     // OpenCV: ya = tyf - ty1

  // Handle border cases properly
  // For pixels outside tile boundaries, use border extrapolation
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
    fx = fminf(fmaxf(fx, 0.f), 1.f);
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
    fy = fminf(fmaxf(fy, 0.f), 1.f);
  }

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
  float enhanced_L = enhanced_L_u8 * 100.0f / 255.0f;  // Scale back to [0,100] range

  // Convert LAB back to RGB
  uint8_t new_r, new_g, new_b;
  lab_to_rgb(enhanced_L, orig_a, orig_b_lab, &new_r, &new_g, &new_b);

  dst_rgb[base + 0] = new_r;
  dst_rgb[base + 1] = new_g;
  dst_rgb[base + 2] = new_b;
}

extern "C" void LaunchApplyLUTBilinearToRGB(const uint8_t *src_rgb, const uint8_t *src_y,
                                            uint8_t *dst_rgb, int H, int W, int tiles_x,
                                            int tiles_y, const uint8_t *luts, cudaStream_t stream) {
  int N = H * W;
  int threads = 512;  // Increased for better memory bandwidth utilization
  int blocks = div_up(N, threads);
  apply_lut_bilinear_rgb_kernel<<<blocks, threads, 0, stream>>>(src_rgb, src_y, dst_rgb, H, W,
                                                                tiles_x, tiles_y, luts);
}

extern "C" void LaunchCLAHE_Grayscale_U8_NHWC(const uint8_t *src_gray, uint8_t *dst_gray, int H,
                                              int W, int tiles_x, int tiles_y, float clip_limit_rel,
                                              unsigned int *tmp_histograms,  // tiles*bins
                                              uint8_t *tmp_luts,             // tiles*bins
                                              cudaStream_t stream) {
  LaunchHistPerTile256(src_gray, H, W, tiles_x, tiles_y, tmp_histograms, stream);
  LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  LaunchApplyLUTBilinearToGray(src_gray, dst_gray, H, W, tiles_x, tiles_y, tmp_luts, stream);
}

extern "C" void LaunchCLAHE_RGB_U8_NHWC(const uint8_t *src_rgb, uint8_t *dst_rgb,
                                        uint8_t *y_plane,  // [H*W]
                                        int H, int W, int tiles_x, int tiles_y,
                                        float clip_limit_rel,
                                        unsigned int *tmp_histograms,  // tiles*bins
                                        uint8_t *tmp_luts,             // tiles*bins
                                        cudaStream_t stream) {
  LaunchRGBToYUint8NHWC(src_rgb, y_plane, H, W, stream);
  LaunchHistPerTile256(y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
  LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  LaunchApplyLUTBilinearToRGB(src_rgb, y_plane, dst_rgb, H, W, tiles_x, tiles_y, tmp_luts, stream);
  CUDA_CHECK(cudaGetLastError());
}

// Optimized version using fused RGB->Y + histogram kernel
extern "C" void LaunchCLAHE_RGB_U8_NHWC_Optimized(const uint8_t *src_rgb, uint8_t *dst_rgb,
                                                  uint8_t *y_plane,  // [H*W]
                                                  int H, int W, int tiles_x, int tiles_y,
                                                  float clip_limit_rel,
                                                  unsigned int *tmp_histograms,  // tiles*bins
                                                  uint8_t *tmp_luts,             // tiles*bins
                                                  cudaStream_t stream) {
  // Fused RGB->Y conversion + histogram computation (saves one kernel launch + memory round-trip)
  LaunchFusedRGBToYHist(src_rgb, y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
  LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
  LaunchApplyLUTBilinearToRGB(src_rgb, y_plane, dst_rgb, H, W, tiles_x, tiles_y, tmp_luts, stream);
  CUDA_CHECK(cudaGetLastError());
}
