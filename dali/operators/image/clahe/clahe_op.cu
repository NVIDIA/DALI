// clahe_op.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t __err = (expr);                                            \
        if (__err != cudaSuccess)                                              \
        {                                                                      \
            printf("CUDA error %d at %s:%d: %s\n",                             \
                   (int)__err, __FILE__, __LINE__, cudaGetErrorString(__err)); \
        }                                                                      \
    } while (0)
#endif

static inline __host__ __device__ int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

// -------------------------------------------------------------------------------------
// Kernel 1: RGB -> Y (uint8). NHWC input (uint8), Y in [0..255] as uint8.
// BT.601 luma: Y = 0.299 R + 0.587 G + 0.114 B
// -------------------------------------------------------------------------------------
__global__ void rgb_to_y_u8_nhwc_kernel(const uint8_t *__restrict__ rgb,
                                        uint8_t *__restrict__ y_out,
                                        int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = H * W;
    if (idx >= N)
        return;

    int c0 = 3 * idx;
    float r = rgb[c0 + 0];
    float g = rgb[c0 + 1];
    float b = rgb[c0 + 2];

    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    int yi = (int)lrintf(fminf(fmaxf(y, 0.f), 255.f));
    y_out[idx] = (uint8_t)yi;
}

// Vectorized version for better memory bandwidth (processes 4 pixels at once)
__global__ void rgb_to_y_u8_nhwc_vectorized_kernel(const uint8_t *__restrict__ rgb,
                                                   uint8_t *__restrict__ y_out,
                                                   int H, int W)
{
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int N = H * W;

    // Process 4 pixels per thread for better memory coalescing
    for (int i = 0; i < 4 && (base_idx + i) < N; ++i)
    {
        int idx = base_idx + i;
        int c0 = 3 * idx;

        float r = rgb[c0 + 0];
        float g = rgb[c0 + 1];
        float b = rgb[c0 + 2];

        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        int yi = (int)lrintf(fminf(fmaxf(y, 0.f), 255.f));
        y_out[idx] = (uint8_t)yi;
    }
}

extern "C" void LaunchRGBToYUint8NHWC(const uint8_t *in_rgb,
                                      uint8_t *y_plane,
                                      int H, int W,
                                      cudaStream_t stream)
{
    int N = H * W;

    // Use vectorized version for better memory bandwidth on larger images
    if (N >= 4096)
    {                      // Threshold for using vectorized version
        int threads = 256; // Each thread processes 4 pixels
        int blocks = div_up(N, threads * 4);
        rgb_to_y_u8_nhwc_vectorized_kernel<<<blocks, threads, 0, stream>>>(in_rgb, y_plane, H, W);
    }
    else
    {
        int threads = 512; // Standard version
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
                                           uint8_t *__restrict__ y_out,
                                           int H, int W,
                                           int tiles_x, int tiles_y,
                                           unsigned int *__restrict__ histograms)
{
    extern __shared__ unsigned int shist[]; // 256 bins
    const int bins = 256;

    int tx = blockIdx.x; // tile x
    int ty = blockIdx.y; // tile y
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
    for (int i = threadIdx.x; i < area; i += blockDim.x)
    {
        int dy = i / (x1 - x0);
        int dx = i - dy * (x1 - x0);
        int x = x0 + dx;
        int y = y0 + dy;

        int pixel_idx = y * W + x;
        int rgb_idx = 3 * pixel_idx;

        // RGB to Y conversion
        float r = rgb[rgb_idx + 0];
        float g = rgb[rgb_idx + 1];
        float b = rgb[rgb_idx + 2];
        float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
        uint8_t y_u8 = (uint8_t)lrintf(fminf(fmaxf(y_val, 0.f), 255.f));

        // Store Y value
        y_out[pixel_idx] = y_u8;

        // Add to histogram
        atomicAdd(&shist[(int)y_u8], 1u);
    }
    __syncthreads();

    // Write back histogram to global memory
    unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        g_hist[i] = shist[i];
    }
}

extern "C" void LaunchFusedRGBToYHist(const uint8_t *rgb,
                                      uint8_t *y_plane,
                                      int H, int W,
                                      int tiles_x, int tiles_y,
                                      unsigned int *histograms,
                                      cudaStream_t stream)
{
    dim3 grid(tiles_x, tiles_y, 1);
    int threads = 512; // Optimized for both compute and shared memory
    size_t shmem = 256 * sizeof(unsigned int);
    fused_rgb_to_y_hist_kernel<<<grid, threads, shmem, stream>>>(
        rgb, y_plane, H, W, tiles_x, tiles_y, histograms);
}

// -------------------------------------------------------------------------------------
// Kernel 2: Histograms per tile (256 bins, uint32).
// One thread block per tile; shared 256-bin histogram.
// Each block sweeps its tile with a grid-stride loop.
// -------------------------------------------------------------------------------------
__global__ void hist_per_tile_256_kernel(const uint8_t *__restrict__ y_plane,
                                         int H, int W,
                                         int tiles_x, int tiles_y,
                                         unsigned int *__restrict__ histograms)
{
    extern __shared__ unsigned int shist[]; // 256 bins
    const int bins = 256;

    int tx = blockIdx.x; // tile x
    int ty = blockIdx.y; // tile y
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
    for (int i = threadIdx.x; i < area; i += blockDim.x)
    {
        int dy = i / (x1 - x0);
        int dx = i - dy * (x1 - x0);
        int x = x0 + dx;
        int y = y0 + dy;
        uint8_t v = y_plane[y * W + x];
        atomicAdd(&shist[(int)v], 1u);
    }
    __syncthreads();

    // Write back to global memory
    unsigned int *g_hist = histograms + (ty * tiles_x + tx) * bins;
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        g_hist[i] = shist[i];
    }
}

extern "C" void LaunchHistPerTile256(const uint8_t *y_plane,
                                     int H, int W,
                                     int tiles_x, int tiles_y,
                                     unsigned int *histograms,
                                     cudaStream_t stream)
{
    dim3 grid(tiles_x, tiles_y, 1);
    int threads = 512; // Increased for better shared memory utilization
    size_t shmem = 256 * sizeof(unsigned int);
    hist_per_tile_256_kernel<<<grid, threads, shmem, stream>>>(
        y_plane, H, W, tiles_x, tiles_y, histograms);
}

// -------------------------------------------------------------------------------------
// Kernel 3: Clip + CDF -> LUT per tile (uint8 LUT).
// clip_limit_rel: relative multiplier of the average bin count per tile
//   limit = clip_limit_rel * (tile_area / bins)
// Excess is redistributed uniformly.
// LUT[v] = round( (cdf[v] - cdf_min) / (tile_area - cdf_min) * 255 )
// -------------------------------------------------------------------------------------
__global__ void clip_cdf_lut_256_kernel(unsigned int *__restrict__ histograms,
                                        int tiles_x, int tiles_y,
                                        int tile_w, int tile_h, // nominal, last tiles smaller
                                        int W, int H,
                                        float clip_limit_rel,
                                        uint8_t *__restrict__ luts)
{
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

    // Compute clip limit (relative to avg bin count)
    float avg = (float)area / bins;
    unsigned int limit = (unsigned int)floorf(clip_limit_rel * avg);

    // Clip and accumulate excess
    __shared__ unsigned int excess;
    if (tid == 0)
        excess = 0u;
    __syncthreads();

    for (int i = tid; i < bins; i += blockDim.x)
    {
        unsigned int v = h[i];
        if (v > limit)
        {
            unsigned int over = v - limit;
            h[i] = limit;
            atomicAdd(&excess, over);
        }
    }
    __syncthreads();

    // Redistribute excess uniformly
    unsigned int add = excess / bins;
    unsigned int rem = excess % bins;
    for (int i = tid; i < bins; i += blockDim.x)
    {
        h[i] += add;
    }
    __syncthreads();
    // Distribute remainder one by one (first 'rem' bins)
    for (int i = tid; i < (int)rem; i += blockDim.x)
    {
        atomicAdd(&h[i], 1u);
    }
    __syncthreads();

    // Prefix-sum (CDF)
    if (tid == 0)
    {
        unsigned int acc = 0u;
        for (int i = 0; i < bins; ++i)
        {
            acc += h[i];
            cdf[i] = acc;
        }
    }
    __syncthreads();

    // Find cdf_min (first non-zero)
    __shared__ unsigned int cdf_min;
    if (tid == 0)
    {
        unsigned int m = 0u;
        for (int i = 0; i < bins; ++i)
        {
            if (cdf[i] != 0u)
            {
                m = cdf[i];
                break;
            }
        }
        cdf_min = m;
    }
    __syncthreads();

    // Build LUT
    uint8_t *lut = luts + (ty * tiles_x + tx) * bins;
    float denom = (float)(area - cdf_min);
    denom = (denom <= 0.f) ? 1.f : denom;

    for (int i = tid; i < bins; i += blockDim.x)
    {
        float val = (float)(cdf[i] - cdf_min) / denom;
        int outv = (int)lrintf(fminf(fmaxf(val, 0.f), 1.f) * 255.f);
        lut[i] = (uint8_t)outv;
    }
}

extern "C" void LaunchClipCdfToLut256(unsigned int *histograms,
                                      int H, int W,
                                      int tiles_x, int tiles_y,
                                      float clip_limit_rel,
                                      uint8_t *luts,
                                      cudaStream_t stream)
{
    int tile_w = div_up(W, tiles_x);
    int tile_h = div_up(H, tiles_y);
    dim3 grid(tiles_x, tiles_y, 1);
    int threads = 512; // Increased for better compute utilization
    clip_cdf_lut_256_kernel<<<grid, threads, 0, stream>>>(
        histograms, tiles_x, tiles_y, tile_w, tile_h, W, H, clip_limit_rel, luts);
}

// -------------------------------------------------------------------------------------
// Kernel 4a: Apply LUT with bilinear interpolation for GRAYSCALE output.
// For each pixel, look up 4 neighboring tile LUTs and bilinearly blend.
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_gray_kernel(const uint8_t *__restrict__ src_y,
                                               uint8_t *__restrict__ dst_y,
                                               int H, int W,
                                               int tiles_x, int tiles_y,
                                               const uint8_t *__restrict__ luts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = H * W;
    if (idx >= N)
        return;

    int y = idx / W;
    int x = idx - y * W;

    // Tile geometry
    int tile_w = div_up(W, tiles_x);
    int tile_h = div_up(H, tiles_y);

    // Tile coordinates
    float gx = (x + 0.5f) / tile_w - 0.5f; // tile-space x
    float gy = (y + 0.5f) / tile_h - 0.5f; // tile-space y
    int tx = (int)floorf(gx);
    int ty = (int)floorf(gy);
    float fx = gx - tx;
    float fy = gy - ty;

    // Handle border cases properly
    // For pixels outside tile boundaries, use border extrapolation
    int tx0, ty0, tx1, ty1;

    if (tx < 0)
    {
        tx0 = tx1 = 0;
        fx = 0.f;
    }
    else if (tx >= tiles_x - 1)
    {
        tx0 = tx1 = tiles_x - 1;
        fx = 0.f;
    }
    else
    {
        tx0 = tx;
        tx1 = tx + 1;
        fx = fminf(fmaxf(fx, 0.f), 1.f);
    }

    if (ty < 0)
    {
        ty0 = ty1 = 0;
        fy = 0.f;
    }
    else if (ty >= tiles_y - 1)
    {
        ty0 = ty1 = tiles_y - 1;
        fy = 0.f;
    }
    else
    {
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

    int outi = (int)lrintf(fminf(fmaxf(v_out, 0.f), 255.f));
    dst_y[idx] = (uint8_t)outi;
}

extern "C" void LaunchApplyLUTBilinearToGray(const uint8_t *src_gray,
                                             uint8_t *dst_gray,
                                             int H, int W,
                                             int tiles_x, int tiles_y,
                                             const uint8_t *luts,
                                             cudaStream_t stream)
{
    int N = H * W;
    int threads = 512; // Increased for better memory bandwidth utilization
    int blocks = div_up(N, threads);
    apply_lut_bilinear_gray_kernel<<<blocks, threads, 0, stream>>>(
        src_gray, dst_gray, H, W, tiles_x, tiles_y, luts);
}

// -------------------------------------------------------------------------------------
// Kernel 4b: Apply LUT for RGB by equalizing Y and rescaling RGB channels.
// Gain approach: gain = Y_eq / max(Y, 1); outRGB = clamp(srcRGB * gain).
// -------------------------------------------------------------------------------------
__global__ void apply_lut_bilinear_rgb_kernel(const uint8_t *__restrict__ src_rgb,
                                              const uint8_t *__restrict__ src_y, // original Y
                                              uint8_t *__restrict__ dst_rgb,
                                              int H, int W,
                                              int tiles_x, int tiles_y,
                                              const uint8_t *__restrict__ luts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = H * W;
    if (idx >= N)
        return;

    int y = idx / W;
    int x = idx - y * W;

    // Tile geometry
    int tile_w = div_up(W, tiles_x);
    int tile_h = div_up(H, tiles_y);

    // Tile coordinates
    float gx = (x + 0.5f) / tile_w - 0.5f;
    float gy = (y + 0.5f) / tile_h - 0.5f;
    int tx = (int)floorf(gx);
    int ty = (int)floorf(gy);
    float fx = gx - tx;
    float fy = gy - ty;

    // Handle border cases properly
    // For pixels outside tile boundaries, use border extrapolation
    int tx0, ty0, tx1, ty1;

    if (tx < 0)
    {
        tx0 = tx1 = 0;
        fx = 0.f;
    }
    else if (tx >= tiles_x - 1)
    {
        tx0 = tx1 = tiles_x - 1;
        fx = 0.f;
    }
    else
    {
        tx0 = tx;
        tx1 = tx + 1;
        fx = fminf(fmaxf(fx, 0.f), 1.f);
    }

    if (ty < 0)
    {
        ty0 = ty1 = 0;
        fy = 0.f;
    }
    else if (ty >= tiles_y - 1)
    {
        ty0 = ty1 = tiles_y - 1;
        fy = 0.f;
    }
    else
    {
        ty0 = ty;
        ty1 = ty + 1;
        fy = fminf(fmaxf(fy, 0.f), 1.f);
    }

    int bins = 256;
    const uint8_t *lut_tl = luts + (ty0 * tiles_x + tx0) * bins;
    const uint8_t *lut_tr = luts + (ty0 * tiles_x + tx1) * bins;
    const uint8_t *lut_bl = luts + (ty1 * tiles_x + tx0) * bins;
    const uint8_t *lut_br = luts + (ty1 * tiles_x + tx1) * bins;

    uint8_t y0 = src_y[idx];
    float v_tl = lut_tl[y0];
    float v_tr = lut_tr[y0];
    float v_bl = lut_bl[y0];
    float v_br = lut_br[y0];

    float v_top = v_tl * (1.f - fx) + v_tr * fx;
    float v_bot = v_bl * (1.f - fx) + v_br * fx;
    float y_eq = v_top * (1.f - fy) + v_bot * fy;

    float gain = y_eq / fmaxf((float)y0, 1.f);

    int base = 3 * idx;
    float r = src_rgb[base + 0] * gain;
    float g = src_rgb[base + 1] * gain;
    float b = src_rgb[base + 2] * gain;

    r = fminf(fmaxf(r, 0.f), 255.f);
    g = fminf(fmaxf(g, 0.f), 255.f);
    b = fminf(fmaxf(b, 0.f), 255.f);

    dst_rgb[base + 0] = (uint8_t)lrintf(r);
    dst_rgb[base + 1] = (uint8_t)lrintf(g);
    dst_rgb[base + 2] = (uint8_t)lrintf(b);
}

extern "C" void LaunchApplyLUTBilinearToRGB(const uint8_t *src_rgb,
                                            const uint8_t *src_y,
                                            uint8_t *dst_rgb,
                                            int H, int W,
                                            int tiles_x, int tiles_y,
                                            const uint8_t *luts,
                                            cudaStream_t stream)
{
    int N = H * W;
    int threads = 512; // Increased for better memory bandwidth utilization
    int blocks = div_up(N, threads);
    apply_lut_bilinear_rgb_kernel<<<blocks, threads, 0, stream>>>(
        src_rgb, src_y, dst_rgb, H, W, tiles_x, tiles_y, luts);
}

extern "C" void LaunchCLAHE_Grayscale_U8_NHWC(const uint8_t *src_gray,
                                              uint8_t *dst_gray,
                                              int H, int W,
                                              int tiles_x, int tiles_y,
                                              float clip_limit_rel,
                                              unsigned int *tmp_histograms, // tiles*bins
                                              uint8_t *tmp_luts,            // tiles*bins
                                              cudaStream_t stream)
{
    LaunchHistPerTile256(src_gray, H, W, tiles_x, tiles_y, tmp_histograms, stream);
    LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
    LaunchApplyLUTBilinearToGray(src_gray, dst_gray, H, W, tiles_x, tiles_y, tmp_luts, stream);
}

extern "C" void LaunchCLAHE_RGB_U8_NHWC(const uint8_t *src_rgb,
                                        uint8_t *dst_rgb,
                                        uint8_t *y_plane, // [H*W]
                                        int H, int W,
                                        int tiles_x, int tiles_y,
                                        float clip_limit_rel,
                                        unsigned int *tmp_histograms, // tiles*bins
                                        uint8_t *tmp_luts,            // tiles*bins
                                        cudaStream_t stream)
{
    LaunchRGBToYUint8NHWC(src_rgb, y_plane, H, W, stream);
    LaunchHistPerTile256(y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
    LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
    LaunchApplyLUTBilinearToRGB(src_rgb, y_plane, dst_rgb, H, W, tiles_x, tiles_y, tmp_luts, stream);
    CUDA_CHECK(cudaGetLastError());
}

// Optimized version using fused RGB->Y + histogram kernel
extern "C" void LaunchCLAHE_RGB_U8_NHWC_Optimized(const uint8_t *src_rgb,
                                                  uint8_t *dst_rgb,
                                                  uint8_t *y_plane, // [H*W]
                                                  int H, int W,
                                                  int tiles_x, int tiles_y,
                                                  float clip_limit_rel,
                                                  unsigned int *tmp_histograms, // tiles*bins
                                                  uint8_t *tmp_luts,            // tiles*bins
                                                  cudaStream_t stream)
{
    // Fused RGB->Y conversion + histogram computation (saves one kernel launch + memory round-trip)
    LaunchFusedRGBToYHist(src_rgb, y_plane, H, W, tiles_x, tiles_y, tmp_histograms, stream);
    LaunchClipCdfToLut256(tmp_histograms, H, W, tiles_x, tiles_y, clip_limit_rel, tmp_luts, stream);
    LaunchApplyLUTBilinearToRGB(src_rgb, y_plane, dst_rgb, H, W, tiles_x, tiles_y, tmp_luts, stream);
    CUDA_CHECK(cudaGetLastError());
}
