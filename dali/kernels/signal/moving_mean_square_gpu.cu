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

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/util.h"
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/kernels/signal/moving_mean_square_gpu.h"

namespace dali {
namespace kernels {
namespace signal {

namespace {

template <typename Out, typename In>
struct SampleDesc {
  Out *out;
  const In *in;
  int64_t len;
};

/**
 * @brief Shared memory access pattern to avoid bank conflicts.
 * @remarks See Example 39-4 from
 *  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */
struct conflict_free_pos {
  static constexpr int kSharedMemBanks = 32;
  static constexpr int kLogMemBanks = 5;

  DALI_HOST_DEV DALI_FORCEINLINE int operator()(int pos) const noexcept {
    return pos + (pos >> kLogMemBanks);
  }
};

struct square {
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE T operator()(T x) const noexcept {
    return x * x;
  }
};

struct divide {
  divide() = default;

  constexpr DALI_HOST_DEV explicit divide(float divisor) : factor(1.0f / divisor) {}

  DALI_HOST_DEV DALI_FORCEINLINE float operator()(float x) const noexcept {
    return x * factor;
  }

  float factor;  // not-initialized in purpose so that it stays trivially constructible.
};


/**
 * @brief Computes the prefix sum (exclusive scan algorithm) in-place on shared memory
 * @remarks Work-efficient algorithm from
 *  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 *
 * @tparam T Data type
 * @tparam SharedMemPos shared memory access pattern (Example 39-3 in the above link)
 * @param buffer Input/Output buffer
 * @param pow2 Size of the buffer (must be a power of 2)
 * @param shm_pos Shared memory access pattern
 */
template <typename T, typename SharedMemPos = conflict_free_pos>
__device__ void PrefixSumSharedMem(T *buffer, int pow2, SharedMemPos shm_pos = {}) {
  int offset = 1;
  int tid = threadIdx.x;

  // build sum in place up the tree
  for (int d = pow2 >> 1; d > 0; d >>= 1) {
    __syncthreads();
    for (int idx = tid; idx < d; idx += blockDim.x) {
      int ai = offset * (2 * idx + 1) - 1;
      int bi = offset * (2 * idx + 2) - 1;
      buffer[shm_pos(bi)] += buffer[shm_pos(ai)];
    }
    offset <<= 1;
  }

  // clear the last element
  if (tid == 0) {
    int last = pow2 - 1;
    buffer[shm_pos(last)] = 0;
  }

  // traverse down tree & build scan
  for (int d = 1; d < pow2; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    for (int idx = tid; idx < d; idx += blockDim.x) {
      int shm_pos_ai = shm_pos(offset * (2 * idx + 1) - 1);
      int shm_pos_bi = shm_pos(offset * (2 * idx + 2) - 1);
      auto t = buffer[shm_pos_ai];
      buffer[shm_pos_ai] = buffer[shm_pos_bi];
      buffer[shm_pos_bi] += t;
    }
  }
  __syncthreads();
}

/**
 * @brief Calculates a running sum of a 1D signal using a sliding window of an arbitrary size.
 *
 * The implementation computes the output on a shared memory buffer of size `logical_block + window`
 * which corresponds to an output region of `logical_block` size.
 *
 * The kernel assumes the same output size as the input, and it pads with zeros at the beginning of
 * the signal so that we can calculate the same number of windows as elements in the input.
 * The input is NOT padded at the end to match every possible window overlap, since we are interested in
 * having an output of the same size as the input.
 *
 * @tparam Out Output data type
 * @tparam In Input data type
 * @tparam Preprocessor Optional preprocessor step (e.g. square)
 * @tparam Postprocessor Optional postprocessing step (e.g. division by window to get running average)
 * @tparam SharedMemPos Shared memory access pattern. Default is choosen to avoid bank conflicts
 * @param samples Sample descriptors
 * @param logical_block Logical block size
 * @param window Window size
 * @param pow2 next power of two of `window + logical_block`
 * @param pre Preprocessing step
 * @param post Postprocessing step
 * @param shm_pos shared memory access pattern
 */
template <typename Out, typename In, typename Preprocessor = dali::identity,
          typename Postprocessor = dali::identity, typename SharedMemPos = conflict_free_pos>
__global__ void SlidingWindowSum(const SampleDesc<Out, In> *samples, int64_t logical_block,
                                 int window, int pow2, Preprocessor pre = {},
                                 Postprocessor post = {}, SharedMemPos shm_pos = {}) {
  extern __shared__ char shm[];  // allocated on invocation
  auto *temp = reinterpret_cast<acc_t<In> *>(shm);

  int sample_idx = blockIdx.y;

  auto &sample = samples[sample_idx];
  Out *output = sample.out;
  const In *input = sample.in;
  int64_t sample_len = sample.len;
  int64_t grid_stride = gridDim.x * logical_block;

  // Each CUDA block calculates the output for `logical_block` samples, where `logical_block` is
  // typically larger than the CUDA block.
  for (int64_t logical_block_start = logical_block * blockIdx.x; logical_block_start < sample_len;
       logical_block_start += grid_stride) {
    const In *logical_block_in_ptr = input + logical_block_start;
    Out *logical_block_out_ptr = output + logical_block_start;
    int64_t logical_block_sz = cuda_min(logical_block, sample_len - logical_block_start);

    const In *extended_blk_start = logical_block_in_ptr - window;
    const In *extended_blk_end = logical_block_in_ptr + logical_block_sz;

    // Step 1: Load extended logical block to shared mem.
    // Out of bounds values are assumed to be 0.
    for (int pos = threadIdx.x; pos < pow2; pos += blockDim.x) {
      acc_t<In> value(0);
      auto extended_blk_ptr = extended_blk_start + pos;
      if (extended_blk_ptr >= input && extended_blk_ptr < extended_blk_end) {
        value = *extended_blk_ptr;
      }
      temp[shm_pos(pos)] = pre(value);
    }

    // // Step 2: Calculate prefix sum of the extended block, in place
    // // (note: __syncthreads already happens inside)
    PrefixSumSharedMem(temp, pow2, shm_pos);

    // Step 3: Compute the output, the sum in window, by subtracting two values of the prefix sum
    // and adding the input value at the current position.
    for (int pos = threadIdx.x; pos < logical_block_sz; pos += blockDim.x) {
      acc_t<In> x = logical_block_in_ptr[pos];
      acc_t<In> out_val = pre(x)                          // current element
                          + temp[shm_pos(window + pos)]   // prefix sum @ pos
                          - temp[shm_pos(pos + 1)];       // prefix sum @ pos - (window-1)
      logical_block_out_ptr[pos] = ConvertSat<Out>(post(out_val));
    }
  }
}

}  // namespace

template <typename InputType>
KernelRequirements MovingMeanSquareGpu<InputType>::Setup(KernelContext &ctx,
                                                         const InListGPU<InputType, 1> &in) {
  KernelRequirements req;
  req.output_shapes = {in.shape};
  return req;
}


template <typename InputType>
void MovingMeanSquareGpu<InputType>::Run(KernelContext &ctx, const OutListGPU<float, 1> &out,
                                         const InListGPU<InputType, 1> &in,
                                         const MovingMeanSquareArgs &args) {
  int nsamples = in.shape.num_samples();
  auto *sample_descs_cpu = ctx.scratchpad->AllocatePinned<SampleDesc<float, InputType>>(nsamples);

  int64_t max_len = 0;
  for (int i = 0; i < nsamples; i++) {
    auto &sample = sample_descs_cpu[i];
    sample.out = out[i].data;
    sample.in = in[i].data;
    sample.len = in[i].shape.num_elements();
    max_len = std::max(sample.len, max_len);
  }
  auto *sample_descs_gpu =
      ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

  conflict_free_pos shm_pos;
  constexpr int kSharedMemBanks = conflict_free_pos::kSharedMemBanks;

  int window_len = args.window_size;

  int max_shm_bytes = GetSharedMemPerBlock();
  int max_shm_elems = max_shm_bytes / sizeof(acc_t<InputType>);

  // Get a power of two that doesn't exceed the desired maximum shared memory size
  int pow2 = prev_pow2(max_shm_elems * kSharedMemBanks / (kSharedMemBanks + 1));

  // If reset interval is given, selects the logical block so that is close to it
  // effectively clearing accummulation error every `reset_interval` samples.
  if (needs_reset<InputType>::value && args.reset_interval > 0 && args.reset_interval < pow2) {
    auto p = prev_pow2(args.reset_interval);
    auto n = next_pow2(args.reset_interval);
    if (p > window_len)
      pow2 = p;
    else if (n < pow2)
      pow2 = n;
  }

  int shm_sz = shm_pos(pow2) * sizeof(acc_t<InputType>);
  int logical_block = pow2 - window_len;
  // Note: logical_block==1 is very wasteful, but better than failing
  assert(logical_block > 0);

  // At the very least we should be able to fit a window plus one element in shared mem
  if (shm_sz > max_shm_bytes) {
    throw std::runtime_error(
      "Can't compute the requested running sum, due to shared memory restrictions");
  }

  dim3 grid(std::min<int64_t>(1024, div_ceil(max_len, 32)), nsamples);
  int block_sz = 512;
  // For mean square we square as a pre-step and divide by the window length at the end
  square pre;
  divide post(window_len);
  SlidingWindowSum<float, InputType><<<grid, block_sz, shm_sz, ctx.gpu.stream>>>(
      sample_descs_gpu, logical_block, window_len, pow2, pre, post, shm_pos);

  CUDA_CALL(cudaGetLastError());
}

template class MovingMeanSquareGpu<double>;
template class MovingMeanSquareGpu<float>;
template class MovingMeanSquareGpu<uint8_t>;
template class MovingMeanSquareGpu<int8_t>;
template class MovingMeanSquareGpu<uint16_t>;
template class MovingMeanSquareGpu<int16_t>;
template class MovingMeanSquareGpu<uint32_t>;
template class MovingMeanSquareGpu<int32_t>;
template class MovingMeanSquareGpu<uint64_t>;
template class MovingMeanSquareGpu<int64_t>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
