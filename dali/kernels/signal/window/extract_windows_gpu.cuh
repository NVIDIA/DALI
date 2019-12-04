// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/signal/window/extract_windows_args.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

namespace window {

constexpr int kBlock = 32;

/// @brief Extract and store kBlock values from kBlock windows.
///
/// This function reads kBlock elements from kBlock windows to shared memory
/// and stores the transposed result to output in columns.
///
/// @remarks This function must be executed by all (or no) threads in a block!
template <int num_pages = 1, typename Dst, typename Src>
__device__ void ExtractWindowsBlock(
    int first_window_idx,
    Dst *__restrict__ dst, ptrdiff_t num_windows, ptrdiff_t stride,
    const Src *__restrict__ src, ptrdiff_t length,
    const float *window,
    int win_len,
    int win_center,
    int step,
    bool reflect,
    int page = 0) {
  __shared__ float tmp[num_pages][kBlock][kBlock+1];  // +1 to avoid bank conflicts
  ptrdiff_t in_window_idx = first_window_idx + threadIdx.y;
  ptrdiff_t window_start = in_window_idx * step - win_center;

  int window_offset = blockIdx.y * kBlock + threadIdx.x;
  if (window_offset < win_len) {
    float w = window ? window[window_offset] : 1;
    ptrdiff_t idx = window_start + window_offset;
    if (reflect) {
      for (;;) {
        if (idx < 0)
          idx = -idx;
        else if (idx >= length)
          idx = 2 * length - 2 - idx;
        else
          break;
      }
    }
    float v = idx >= 0 && idx < length ? ConvertNorm<float>(src[idx]) * w : Src();
    tmp[page][threadIdx.y][threadIdx.x] = v;
  }
  __syncthreads();

  int win_ofs = blockIdx.y * kBlock + threadIdx.y;
  ptrdiff_t win_idx = first_window_idx + threadIdx.x;

  if (win_ofs < win_len && win_idx < num_windows)
    dst[stride * win_ofs + win_idx] = ConvertSatNorm<Dst>(tmp[page][threadIdx.x][threadIdx.y]);
}


template <typename Dst, typename Src>
__global__ void ExtractWindowsKernel(
    Dst *__restrict__ dst, ptrdiff_t num_windows, ptrdiff_t stride,
    const Src *__restrict__ src, ptrdiff_t length,
    const float *window,
    int win_len,
    int win_center,
    int step,
    bool reflect) {
  // This kernel reads kBlock elements from kBlock windows to shared memory
  // and stores the transposed result to output in columns.

  ExtractWindowsBlock(
    blockIdx.x * kBlock,        // first window index
    dst, num_windows, stride,   // output
    src, length,                // input
    window, win_len, win_center, step, reflect);  // windowing options
}

struct SampleDesc {
  /// @brief Pointer to the beginning of the output window buffer
  void *__restrict__ output;
  /// @brief Total number of windows to extract from this signal
  int num_windows;
  /// @brief Stride between consecutive samples in a window (typically num_windows)
  ptrdiff_t output_stride;
  /// @brief Pointer to the beginning of the signal
  const void *__restrict__ input;
  /// @brief Length, in samples, of the input signal
  ptrdiff_t length;
};

struct BlockDesc {
  int sample_idx;
  /// @brief Index of the first window in this block.
  ///
  /// The index of the last block can be calculated by the kernel and doesn't need to be
  /// passed explicitly.
  int start;
};

/**
 * @brief Batched block extraction kernel
 *
 * CUDA block size is fixed kBlock x kBlock - it processes kBlock values from kBlock windows.
 * A logical block may contain multiple CUDA blocks - the multiplier is always
 * equal to 1 << block_shift.
 */
template <typename Dst, typename Src>
__global__ void ExtractWindowsBatchedKernel(
    const SampleDesc *samples,
    const window::BlockDesc *blocks,
    int windows_per_block,
    const float *window,
    int win_len,
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

  int block_end = min(blk.start + windows_per_block, num_windows);

  // NOTE: This loop is dynamically uniform. Introducing any thread-index dependent condition
  // will break the code, leading to undefined behavior (including hangs).
  for (int pos = blk.start, page = 0; pos < block_end; pos += kBlock, page = 1-page) {
    ExtractWindowsBlock<2>(
      pos,                        // first window index
      dst, num_windows, stride,   // output
      src, length,                // input
      window, win_len, win_center, step, reflect,  // windowing options
      page);  // page-flipped temporary buffer avoids additional __syncthreads
  }
}

}  // namespace window

template <typename Dst, typename Src>
struct ExtractWindowsGpuImpl {
  using SampleDesc = window::SampleDesc;
  using BlockDesc = window::BlockDesc;

  KernelRequirements Setup(
      KernelContext &context,
      const InListGPU<Src, 1> &input,
      const ExtractWindowsArgs &args,
      bool concatenate,
      int out_win_length = -1) {
    block_dim  = dim3(window::kBlock, window::kBlock);

    const auto &lengths = input.shape;

    this->args = args;
    this->concatenate = concatenate;
    if (out_win_length < 0)
      out_win_length = this->args.window_length;
    else if (out_win_length < this->args.window_length)
      this->args.window_length = out_win_length;

    int N = lengths.num_samples();
    int ygrid = div_ceil(this->args.window_length, window::kBlock);

    int64_t total_windows = 0;

    TensorListShape<2> out_shape;
    out_shape.resize(concatenate ? 1 : N);

    windows_per_block = window::kBlock;
    int xgrid = 0;
    for (int i = 0; i < N; i++) {
      int nwin = args.num_windows(lengths[i][0]);
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
    while (xgrid > kMaxBlocks && xgrid > 2 * N) {
      windows_per_block <<= 1;
      xgrid = 0;
      for (int i = 0; i < N; i++) {
        int nwin = args.num_windows(lengths[i][0]);
        xgrid += div_ceil(nwin, windows_per_block);
      }
    }

    grid_dim = dim3(xgrid, ygrid);

    ScratchpadEstimator se;
    se.add<SampleDesc>(AllocType::GPU, N);
    se.add<BlockDesc>(AllocType::GPU, xgrid);
    se.add<SampleDesc>(AllocType::Host, N);
    se.add<BlockDesc>(AllocType::Host, xgrid);

    KernelRequirements req;
    req.scratch_sizes = se.sizes;
    req.output_shapes = { out_shape };

    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Dst, 2> &out,
           const InListGPU<Src, 1> &in,
           const InTensorGPU<float, 1> &window) {
    size_t b = 0;
    int nblocks = grid_dim.x;

    int N = in.num_samples();

    auto *cpu_samples = ctx.scratchpad->Allocate<SampleDesc>(AllocType::Host, N);
    auto *cpu_blocks = ctx.scratchpad->Allocate<BlockDesc>(AllocType::Host, nblocks);

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

    window::ExtractWindowsBatchedKernel<Dst, Src>
    <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(
      gpu_samples, gpu_blocks, windows_per_block,
      window.data, args.window_length, args.window_center,
      args.window_step, args.padding == Padding::Reflect);
  }

  struct BlockInfo {
    const Src *src_base;
    int64_t start, end;
  };
  dim3 block_dim, grid_dim;
  int windows_per_block = window::kBlock;
  ExtractWindowsArgs args;
  bool concatenate = true;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_GPU_CUH_
