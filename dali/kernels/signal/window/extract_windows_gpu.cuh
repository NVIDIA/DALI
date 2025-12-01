// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_CUH_
#define DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_CUH_

#include <cuda_runtime.h>
#include <algorithm>

#include "dali/core/convert.h"
#include "dali/core/boundary.h"
#include "dali/kernels/signal/window/extract_windows_args.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

namespace window {

constexpr int kBlock = 32;


/**
 *  @brief Extract and store kBlock values from kBlock windows.
 * This function reads kBlock elements from kBlock windows to shared memory
 * and stores the transposed result to output in columns.
 *
 * @param first_window_idx  Index of the first window processed by a block
 * @param dst               destination buffer
 * @param num_windows       maximum number of windows to extract from given input
 * @param stride            stride, in elements, between respective samples in consecutive windows
 * @param src               input signal
 * @param length            length, in samples, of the signal
 * @param window            window function
 * @param out_win_len       length, in samples, of the output window;
 *                          if `in_win_len` < `out_win_len`, output is zero-padded
 * @param in_window_start   window start offset within the output window
 * @param in_win_len        length, in samples, of the window
 * @param win_center        center of the window function; typically win_len/2
 * @param step              step, in samples, between consecutive windows in the input
 * @param reflect           if true, reflect the signal at ends, otherwise zero-pad
 * @param page              which shared memory buffer to use; page flipping
 *                          saves one __syncthreads
 * @tparam num_pages        number of shared memory buffers to use; should be 2 if a single CUDA
 *                          block processes multiple chuncks of data, otherwise 1.
 *
 * @remarks This function must be executed by all (or no) threads in a block!
 */
template <int num_pages = 1, typename Dst, typename Src>
__device__ void ExtractVerticalWindowsBlock(
    int first_window_idx,
    Dst *__restrict__ dst, ptrdiff_t num_windows, ptrdiff_t stride,
    const Src *__restrict__ src, ptrdiff_t length,
    const float *__restrict__ window,
    int out_win_len,
    int in_win_start,
    int in_win_len,
    int win_center,
    int step,
    bool reflect,
    int page = 0) {
  __shared__ float tmp[num_pages][kBlock][kBlock+1];  // +1 to avoid bank conflicts
  ptrdiff_t in_window_idx = first_window_idx + threadIdx.y;
  ptrdiff_t window_start = in_window_idx * step - win_center;

  int in_win_offset = blockIdx.y * kBlock + threadIdx.x - in_win_start;
  float v = 0;
  if (in_win_offset >= 0 && in_win_offset < in_win_len) {
    float w = window ? window[in_win_offset] : 1;
    ptrdiff_t idx = window_start + in_win_offset;
    if (reflect) {
      idx = boundary::idx_reflect_101(idx, length);
    }
    v = idx >= 0 && idx < length ? ConvertNorm<float>(src[idx]) * w : 0.0f;
  }
  tmp[page][threadIdx.y][threadIdx.x] = v;
  __syncthreads();

  int win_ofs = blockIdx.y * kBlock + threadIdx.y;
  ptrdiff_t win_idx = first_window_idx + threadIdx.x;

  if (win_ofs < out_win_len && win_idx < num_windows)
    dst[stride * win_ofs + win_idx] = ConvertSatNorm<Dst>(tmp[page][threadIdx.x][threadIdx.y]);
}

/**
 * @brief Copy a sample to the windows containing it, applying the window function.
 *
 * This function takes one sample from the input and copies it to all windows that the sample
 * contributes to.
 * This function is optimized for cases when each sample contributes to the same number of windows.
 *
 * @param idx           Index of the sample processed by current thread, relative to `src`.
 *                      If the index is out of range (<0 or >= length), the signal is reflected
 *                      or padded with zeros.
 * @param dst           destination buffer
 * @param num_windows   maximum number of windows to extract from given input
 * @param stride        stride, in elements, between consecutive windows in `dst`
 * @param src           input signal
 * @param length        length, in samples, of the signal
 * @param window        window function
 * @param win_len       length, in samples, of the window
 * @param win_center    center of the window function; typically win_len/2
 * @param step          step, in samples, between consecutive windows in the input
 * @param reflect       if true, reflect the signal at ends, otherwise zero-pad
 */
template <typename Dst, typename Src>
__device__ void ExtractHorizontalWindows(
    ptrdiff_t idx,
    Dst *__restrict__ dst, ptrdiff_t num_windows, ptrdiff_t stride,
    const Src *__restrict__ src, ptrdiff_t length,
    const float *__restrict__ window,
    int win_len,
    int win_center,
    int step,
    bool reflect) {

  // calculate the index of the first window that sample at `idx ` contributes to
  ptrdiff_t idx0 = idx + win_center - win_len + step;  // add step to round up
  ptrdiff_t win0 = ::max(ptrdiff_t(), idx0/step);

  float value = 0;

  if (reflect) {
    ptrdiff_t src_idx = boundary::idx_reflect_101(idx, length);
    value = ConvertSatNorm<float>(src[src_idx]);
  } else {
    if (idx >= 0 && idx < length)
      value = ConvertSatNorm<float>(src[idx]);
  }

  for (int win_ofs = idx - (win0 * step - win_center), win_idx = win0;
      win_ofs >= 0 && win_idx < num_windows;
      win_ofs -= step, win_idx++) {
    float v = window ? value * window[win_ofs] : value;
    dst[win_idx * stride + win_ofs] = ConvertSatNorm<Dst>(v);
  }
}

template <typename Dst, typename Src>
__global__ void ExtractVerticalWindowsKernel(
    Dst *__restrict__ dst, ptrdiff_t num_windows, ptrdiff_t stride,
    const Src *__restrict__ src, ptrdiff_t length,
    const float *__restrict__ window,
    int out_win_len,
    int in_win_start,
    int in_win_len,
    int win_center,
    int step,
    bool reflect) {
  // This kernel reads kBlock elements from kBlock windows to shared memory
  // and stores the transposed result to output in columns.

  ExtractVerticalWindowsBlock(
    blockIdx.x * kBlock,        // first window index
    dst, num_windows, stride,   // output
    src, length,                // input
    window, out_win_len, in_win_start, in_win_len, win_center, step, reflect);  // windowing options
}

struct SampleDesc {
  /// @brief Pointer to the beginning of the output window buffer
  void *__restrict__ output;
  /// @brief Total number of windows to extract from this signal
  int num_windows;
  /// @brief Stride between samples (for vertical windows) or between windows (for horizontal ones)
  ptrdiff_t output_stride;
  /// @brief Pointer to the beginning of the signal
  const void *__restrict__ input;
  /// @brief Length, in samples, of the input signal
  ptrdiff_t length;
};

struct BlockDesc {
  /// @brief index of the sample (tensor) in the batch
  int sample_idx;
  /// @brief Index of the first window in this block.
  ///
  /// The index of the last block can be calculated by the kernel and doesn't need to be
  /// passed explicitly.
  int start;
};

struct HorizontalBlockDesc {
  /// @brief index of the sample (tensor) in the batch
  int sample_idx;
  /// @brief actual length of the block (may be less than blockDim)
  int count;
  /// @brief Index of the first value in this block
  ptrdiff_t start;
};

/**
 * @brief Batched vertical window extraction kernel
 *
 * Extracts blocks from 1D signal and puts them in the output as vertical slices.
 *
 * CUDA block size is fixed kBlock x kBlock - it processes kBlock values from kBlock windows.
 * A logical block may contain multiple CUDA blocks - the multiplier is always
 * equal to 1 << block_shift.
 */
template <typename Dst, typename Src>
__global__ void ExtractVerticalWindowsBatchedKernel(
    const SampleDesc *samples,
    const BlockDesc *blocks,
    int windows_per_block,
    const float *__restrict__ window,
    int out_win_len,
    int in_win_start,
    int in_win_len,
    int win_center,
    int step,
    bool reflect) {
  int block_idx = blockIdx.x;
  BlockDesc blk = blocks[block_idx];
  SampleDesc sample = samples[blk.sample_idx];
  const Src *__restrict__ src = static_cast<const Src*>(sample.input);
  Dst *__restrict__ dst = static_cast<Dst*>(sample.output);
  ptrdiff_t stride = sample.output_stride;
  int num_windows = sample.num_windows;
  ptrdiff_t length = sample.length;

  int block_end = ::min(blk.start + windows_per_block, num_windows);

  // NOTE: This loop is dynamically uniform. Introducing any thread-index dependent condition
  // will break the code, leading to undefined behavior (including hangs).
  for (int pos = blk.start, page = 0; pos < block_end; pos += kBlock, page = 1-page) {
    ExtractVerticalWindowsBlock<2>(
      pos,                        // first window index
      dst, num_windows, stride,   // output
      src, length,                // input
      window, out_win_len, in_win_start, in_win_len, win_center, step, reflect,  // win options
      page);  // page-flipped temporary buffer avoids additional __syncthreads
  }
}

/**
 * @brief Batched horizontal window extraction kernel
 *
 * Extracts blocks from 1D signal and puts them in the output as horizontal slices.
 *
 * CUDA block size is variable (and generally quite large).
 */
template <typename Dst, typename Src>
__global__ void ExtractHorizontalWindowsBatchedKernel(
    const SampleDesc *samples,
    const HorizontalBlockDesc *blocks,
    const float *__restrict__ window,
    int win_len,
    int win_center,
    int step,
    bool reflect) {
  int block_idx = blockIdx.x;
  HorizontalBlockDesc blk = blocks[block_idx];
  SampleDesc sample = samples[blk.sample_idx];
  const Src *__restrict__ src = static_cast<const Src*>(sample.input);
  Dst *__restrict__ dst = static_cast<Dst*>(sample.output);
  ptrdiff_t stride = sample.output_stride;
  int num_windows = sample.num_windows;
  ptrdiff_t length = sample.length;

  ptrdiff_t block_end = blk.start + blk.count;

  for (ptrdiff_t pos = blk.start + threadIdx.x; pos < block_end; pos += blockDim.x) {
    ExtractHorizontalWindows(
      pos,                        // input offset
      dst, num_windows, stride,   // output
      src, length,                // input
      window, win_len, win_center, step, reflect);  // windowing options
  }
}


struct PadHorizontalWindowsBlock {
  /// @brief Pointer to the first _padding_ element in the block
  void *base;
  /// @brief Number of windows in this block
  int num_windows;
  /// @brief Stride between windows
  int window_stride;
  /// @brief Pad length
  int pad_length;
};


/**
 * @brief Pads windows in non-contiguous output with zeros.
 */
template <typename T>
__global__ void PadHorizontalWindowsKernel(
  const PadHorizontalWindowsBlock *__restrict__ blocks) {
  T *__restrict__ base = static_cast<T *>(blocks[blockIdx.z].base);
  int num_win = blocks[blockIdx.z].num_windows;
  int win_stride = blocks[blockIdx.z].window_stride;
  int pad_length = blocks[blockIdx.z].pad_length;
  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < num_win; y += blockDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < pad_length; x += blockDim.x) {
      base[win_stride * y + x] = 0;
    }
  }
}

}  // namespace window

/**
 * @tparam Dst output type - for integral outputs, the intermediate floating point value is
 *                           assumed to be in -1..1 (for signed Dst) or 0..1 (for unsigned)
 *                           range and scaled to occupy the Dst dynamic range;
 *                           if `Dst` is `float`, the output value is stored as-is.
 * @tparam Src input type - `float` or integer type; the integer ranges are converted to
 *                          floats and normalized to -1..1 range before applying the window
 *                          function.
 */
template <typename Dst, typename Src>
struct ExtractWindowsImplGPU {
  virtual KernelRequirements Setup(
      KernelContext &context,
      span<const int64_t> input_shape,
      const ExtractWindowsArgs &args,
      bool concatenate,
      int out_win_length = -1) = 0;

  virtual void Run(
      KernelContext &ctx,
      const OutListGPU<Dst, 2> &out,
      const InListGPU<Src, 1> &in,
      const InTensorGPU<float, 1> &window) = 0;

  virtual bool IsVertical() const { return false; }

  virtual ~ExtractWindowsImplGPU() = default;
};


/// @brief Extracts windows and stores them in columns
template <typename Dst, typename Src>
struct ExtractVerticalWindowsImplGPU : ExtractWindowsImplGPU<Dst, Src> {
  using SampleDesc = window::SampleDesc;
  using BlockDesc = window::BlockDesc;

  bool IsVertical() const override { return true; }

  KernelRequirements Setup(
      KernelContext &context,
      span<const int64_t> lengths,
      const ExtractWindowsArgs &args,
      bool concatenate,
      int out_win_length) override {
    block_dim  = dim3(window::kBlock, window::kBlock);

    this->args = args;
    this->concatenate = concatenate;
    if (out_win_length < 0)
      out_win_length = this->args.window_length;
    else if (out_win_length < this->args.window_length)
      this->args.window_length = out_win_length;
    this->out_win_length = out_win_length;

    int N = lengths.size();
    int ygrid = div_ceil(out_win_length, window::kBlock);

    int64_t total_windows = 0;

    TensorListShape<2> out_shape;
    out_shape.resize(concatenate ? 1 : N);

    windows_per_block = window::kBlock;
    int xgrid = 0;
    for (int i = 0; i < N; i++) {
      int nwin = args.num_windows(lengths[i]);
      total_windows += nwin;
      int blocks = div_ceil(nwin, windows_per_block);
      xgrid += blocks;
      if (!concatenate) {
        out_shape.set_tensor_shape(i, { out_win_length, nwin });
      }
    }
    if (concatenate) {
      out_shape.set_tensor_shape(0, { out_win_length, total_windows });
    }

    const int kMaxBlocks = 0x10000;
    while (xgrid > kMaxBlocks && xgrid >= 2 * N) {
      windows_per_block <<= 1;
      xgrid = 0;
      for (int i = 0; i < N; i++) {
        int nwin = args.num_windows(lengths[i]);
        xgrid += div_ceil(nwin, windows_per_block);
      }
    }

    grid_dim = dim3(xgrid, ygrid);

    KernelRequirements req;
    req.output_shapes = { out_shape };

    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Dst, 2> &out,
           const InListGPU<Src, 1> &in,
           const InTensorGPU<float, 1> &window) override {
    int b = 0;
    int nblocks = grid_dim.x;

    int N = in.num_samples();

    auto *cpu_samples = ctx.scratchpad->AllocatePinned<SampleDesc>(N);
    auto *cpu_blocks = ctx.scratchpad->AllocatePinned<BlockDesc>(nblocks);

    assert(!window.data || window.shape[0] == args.window_length);

    ptrdiff_t total_windows = 0;
    for (int i = 0; i < N; i++) {
      int out_tensor = concatenate ? 0 : i;
      ptrdiff_t out_width = out.tensor_shape_span(out_tensor)[1];
      ptrdiff_t in_length = in.tensor_shape_span(i)[0];

      int nwindows = args.num_windows(in_length);
      int nblk = div_ceil(nwindows, windows_per_block);
      assert(out_width >= nwindows);

      ptrdiff_t out_offset = concatenate ? total_windows : 0;

      auto &sample = cpu_samples[i];
      sample.output = out.tensor_data(out_tensor) + out_offset;
      sample.num_windows = nwindows;
      sample.output_stride = out_width;
      sample.input = in.tensor_data(i);
      sample.length = in_length;

      for (int w = 0; w < nwindows; w += windows_per_block) {
        assert(b < nblocks);
        cpu_blocks[b++] = { i, w };
      }
      total_windows += nwindows;
    }

    SampleDesc *gpu_samples;
    BlockDesc *gpu_blocks;

    std::tie(gpu_samples, gpu_blocks) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream,
      make_span(cpu_samples, N),
      make_span(cpu_blocks, nblocks));

    int in_win_start =
        out_win_length > args.window_length ? (out_win_length - args.window_length) / 2 : 0;

    window::ExtractVerticalWindowsBatchedKernel<Dst, Src>
    <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(
      gpu_samples, gpu_blocks, windows_per_block,
      window.data, out_win_length, in_win_start, args.window_length, args.window_center,
      args.window_step, args.padding == Padding::Reflect);
    CUDA_CALL(cudaGetLastError());
  }

  dim3 block_dim, grid_dim;
  int windows_per_block = window::kBlock;
  ExtractWindowsArgs args;
  int out_win_length = -1;
  bool concatenate = true;
};

/// @brief Extracts windows and stores them in rows
template <typename Dst, typename Src>
struct ExtractHorizontalWindowsImplGPU : ExtractWindowsImplGPU<Dst, Src> {
  using SampleDesc = window::SampleDesc;
  using BlockDesc = window::HorizontalBlockDesc;
  using PadBlock = window::PadHorizontalWindowsBlock;

  KernelRequirements Setup(
      KernelContext &context,
      span<const int64_t> lengths,
      const ExtractWindowsArgs &args,
      bool concatenate,
      int out_win_length) override {
    this->args = args;
    this->concatenate = concatenate;
    if (out_win_length < 0)
      out_win_length = this->args.window_length;
    else if (out_win_length < this->args.window_length)
      this->args.window_length = out_win_length;

    this->out_win_length = out_win_length;
    int N = lengths.size();

    this->pad_left = (this->out_win_length - this->args.window_length) / 2;
    this->pad_right = this->out_win_length - this->args.window_length - this->pad_left;
    int pad_length = std::max(pad_left, pad_right);

    int64_t total_windows = 0;

    TensorListShape<2> out_shape;
    out_shape.resize(concatenate ? 1 : N);

    constexpr int kDefaultBlockSize = 256;
    logical_block_size = kDefaultBlockSize;
    int64_t max_padded_length = 0;
    int max_win_per_input = 0;
    for (int i = 0; i < N; i++) {
      int64_t length = lengths[i];
      int nwin = args.num_windows(length);
      assert(nwin >= 1);  // the calling operator should have checked that

      int64_t padded_length = static_cast<int64_t>(nwin-1) * args.window_step + args.window_length;
      if (padded_length > max_padded_length)
        max_padded_length = padded_length;

      if (nwin > max_win_per_input)
        max_win_per_input = nwin;
      total_windows += nwin;
      if (!concatenate) {
        out_shape.set_tensor_shape(i, { nwin, out_win_length });
      }
    }
    // for each sample, the nwin >= 1 is expected
    // and the op call should be skipped if there are no samples
    assert(max_win_per_input >= 1);

    if (concatenate) {
      out_shape.set_tensor_shape(0, { total_windows, out_win_length });
    }

    if (max_padded_length <= kDefaultBlockSize) {
      block_dim = logical_block_size = max_padded_length;
      grid_dim = N;
    } else {
      const int kMaxBlocks = 0x10000;
      for (;;) {
        grid_dim = 0;
        for (int i = 0; i < N; i++) {
          ptrdiff_t nwin = args.num_windows(lengths[i]);
          ptrdiff_t length = (nwin-1) * args.window_step + args.window_length;
          grid_dim += div_ceil(length, logical_block_size);
        }
        if (grid_dim <= kMaxBlocks || grid_dim < 2 * N)
          break;
        logical_block_size <<= 1;
      }
      block_dim = kDefaultBlockSize;
    }

    pad_grid = dim3(0, 0, 0);
    pad_block_size = 0;
    if (out_win_length > args.window_length && !concatenate) {
      int max_pad_block_x = 32;
      if (max_win_per_input < 32) {
        max_pad_block_x = 1024/max_win_per_input;
      }
      int pad_block_x = clamp(pad_length, 1, max_pad_block_x);
      int pad_block_y = clamp(1024/pad_block_x, 1, max_win_per_input);
      pad_block_size = pad_block_y;
      const int kMaxBlocks = 0x10000;
      pad_grid.x = div_ceil(pad_length, pad_block_x);
      int pad_blocks = 0;
      for (;;) {
        pad_blocks = 0;
        for (int i = 0; i < N; i++) {
          ptrdiff_t nwin = args.num_windows(lengths[i]);
          pad_blocks += div_ceil(nwin, pad_block_size) * ((pad_left > 0) + (pad_right > 0));
        }
        if (grid_dim <= kMaxBlocks || grid_dim < 2 * N)
          break;
        pad_block_size <<= 1;
      }
      pad_grid.y = div_ceil(pad_block_size, pad_block_y);
      pad_grid.z = pad_blocks;
      pad_block = dim3(pad_block_x, pad_block_y, 1);
    }

    KernelRequirements req;
    req.output_shapes = { out_shape };

    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Dst, 2> &out,
           const InListGPU<Src, 1> &in,
           const InTensorGPU<float, 1> &window) override {
    int b = 0;

    int N = in.num_samples();

    auto *cpu_samples = ctx.scratchpad->AllocatePinned<SampleDesc>(N);
    auto *cpu_blocks = ctx.scratchpad->AllocatePinned<BlockDesc>(grid_dim);
    auto *cpu_pad_blocks = ctx.scratchpad->AllocatePinned<PadBlock>(pad_grid.z);

    assert(!window.data || window.shape[0] == args.window_length);

    ptrdiff_t total_windows = 0;
    int pad_blocks = 0;

    for (int i = 0; i < N; i++) {
      int out_tensor = concatenate ? 0 : i;
      ptrdiff_t out_width = out.tensor_shape_span(out_tensor)[1];
      ptrdiff_t in_length = in.tensor_shape_span(i)[0];
      assert(out_width >= args.window_length);

      int nwindows = args.num_windows(in_length);

      // this is the number of samples covered by this block - it includes padding (if any)
      // but also takes into account that the end of the last window doesn't need to coincide
      // with the end of (padded) signal.
      ptrdiff_t length_covered = (nwindows - 1) * args.window_step + args.window_length;
      int nblk = div_ceil(nwindows, div_ceil(length_covered, logical_block_size));

      ptrdiff_t out_offset = concatenate ? total_windows * out_width : 0;

      auto &sample = cpu_samples[i];
      Dst *out_ptr = out.tensor_data(out_tensor) + out_offset;
      sample.output = out_ptr + pad_left;
      sample.num_windows = nwindows;
      sample.output_stride = out_width;
      sample.input = in.tensor_data(i);
      sample.length = in_length;

      ptrdiff_t start = -args.window_center;
      ptrdiff_t end = start + length_covered;

      for (ptrdiff_t pos = start; pos < end; pos += logical_block_size) {
        assert(b < grid_dim);
        int count = std::min<ptrdiff_t>(end - pos, logical_block_size);
        cpu_blocks[b++] = { i, count, pos };
      }
      total_windows += nwindows;

      // calculate padding setup
      if (pad_grid.z) {
        auto *ptr = out_ptr;
        for (int w = 0; w < nwindows; w += pad_block_size) {
          assert(pad_blocks < static_cast<int>(pad_grid.z));
          int cnt = std::min(pad_block_size, nwindows - w);
          if (pad_left > 0) {
            cpu_pad_blocks[pad_blocks++] = {
              ptr, cnt, static_cast<int>(sample.output_stride), pad_left
            };
          }
          if (pad_right > 0) {
            cpu_pad_blocks[pad_blocks++] = {
              ptr + pad_left + args.window_length, cnt,
              static_cast<int>(sample.output_stride), pad_right
            };
          }
          ptr += cnt * sample.output_stride;
        }
      }
    }
    assert(pad_blocks == static_cast<int>(pad_grid.z));

    SampleDesc *gpu_samples;
    BlockDesc *gpu_blocks;
    PadBlock *gpu_pad_blocks;

    std::tie(gpu_samples, gpu_blocks, gpu_pad_blocks) = ctx.scratchpad->ToContiguousGPU(
      ctx.gpu.stream,
      make_span(cpu_samples, N),
      make_span(cpu_blocks, grid_dim),
      make_span(cpu_pad_blocks, pad_grid.z));

    window::ExtractHorizontalWindowsBatchedKernel<Dst, Src>
    <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(
      gpu_samples, gpu_blocks,
      window.data, args.window_length, args.window_center,
      args.window_step, args.padding == Padding::Reflect);
    CUDA_CALL(cudaGetLastError());

    int padding_length = out_win_length - args.window_length;
    if (padding_length > 0) {
      if (pad_grid.z) {
        window::PadHorizontalWindowsKernel<Dst>
            <<<pad_grid, pad_block, 0, ctx.gpu.stream>>>(gpu_pad_blocks);
      } else {
        assert(concatenate);
        auto stride = out.tensor_shape_span(0)[1];
        if (pad_left > 0) {
          CUDA_CALL(cudaMemset2DAsync(out.tensor_data(0), stride * sizeof(Dst), 0,
                                      pad_left * sizeof(Dst), total_windows, ctx.gpu.stream));
        }
        if (pad_right > 0) {
          CUDA_CALL(cudaMemset2DAsync(out.tensor_data(0) + pad_left + args.window_length,
                                      stride * sizeof(Dst), 0, pad_right * sizeof(Dst),
                                      total_windows, ctx.gpu.stream));
        }
      }
    }
  }

  int block_dim = 0, grid_dim = 0;
  dim3 pad_grid = 0;
  dim3 pad_block = dim3(32, 32, 1);
  int logical_block_size = 0;
  int pad_block_size = 0;
  ExtractWindowsArgs args;
  int out_win_length = -1;
  int pad_left = 0;
  int pad_right = 0;
  bool concatenate = true;
};


}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_CUH_
