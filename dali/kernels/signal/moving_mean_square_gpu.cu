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
#include "dali/core/util.h"
#include "dali/kernels/signal/moving_mean_square.h"
#include "dali/kernels/signal/moving_mean_square_gpu.h"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

namespace dali {
namespace kernels {
namespace signal {

struct SampleDesc {
  void *out;
  const void *in;
  int64_t len;
};

struct conflict_free_pos {
  DALI_HOST_DEV DALI_FORCEINLINE int operator()(int pos) const noexcept {
    return pos + CONFLICT_FREE_OFFSET(pos);
  }
};

struct square {
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE T operator()(T x) const noexcept {
    return x * x;
  }
};

struct divide {
  explicit divide(float divisor) {
    factor_ = 1.0 / divisor;
  }

  DALI_HOST_DEV DALI_FORCEINLINE float operator()(float x) const noexcept {
    return x * factor_;
  }

  float factor_;
};


template <typename T, typename SharedMemPos = conflict_free_pos>
__device__ void PrefixSumSharedMem(T *buffer, int pow2, SharedMemPos shm_pos = {}) {
  int offset = 1;
  int tid = threadIdx.x;
  for (int d = pow2 >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      buffer[shm_pos(bi)] += buffer[shm_pos(ai)];
    }
    offset <<= 1;
  }

  if (tid == 0) {
    int last = pow2 - 1;
    buffer[shm_pos(last)] = 0;  // clear the last element
  }

  for (int d = 1; d < pow2; d <<= 1) {  // traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int shm_pos_ai = shm_pos(offset * (2 * tid + 1) - 1);
      int shm_pos_bi = shm_pos(offset * (2 * tid + 2) - 1);
      auto t = buffer[shm_pos_ai];
      buffer[shm_pos_ai] = buffer[shm_pos_bi];
      buffer[shm_pos_bi] += t;
    }
  }
}

template <typename Out, typename In, typename Preprocessor = dali::identity,
          typename Postprocessor = dali::identity, typename SharedMemPos = conflict_free_pos>
__global__ void SlidingWindowSum(const SampleDesc *samples, int logical_block, int window, int pow2,
                                 Preprocessor pre = {}, Postprocessor post = {},
                                 SharedMemPos shm_pos = {}) {
  extern __shared__ char shm[];  // allocated on invocation
  auto *temp = reinterpret_cast<acc_t<In> *>(shm);

  int sample_idx = blockIdx.y;

  auto &sample = samples[sample_idx];
  Out *output = reinterpret_cast<Out *>(sample.out);
  const In *input = reinterpret_cast<const In *>(sample.in);
  int64_t sample_len = sample.len;
  int64_t grid_stride = gridDim.x * blockDim.x;

  for (int64_t logical_block_start = logical_block * blockIdx.x; logical_block_start < sample_len;
       logical_block_start += grid_stride) {
    int n = cuda_min(static_cast<int>(sample.len - logical_block_start), logical_block);

    const In *logical_block_in_ptr = input + logical_block_start;
    Out *logical_block_out_ptr = output + logical_block_start;

    const In *extended_blk_start = logical_block_in_ptr - window;
    const In *extended_blk_end = logical_block_in_ptr + cuda_min(logical_block, n);
    int extended_blk_sz = logical_block + window;

    // Step 1: Load extended block to shared mem
    for (int pos = threadIdx.x; pos < extended_blk_sz; pos += blockDim.x) {
      acc_t<In> value(0);
      auto extended_blk_ptr = extended_blk_start + pos;
      if (extended_blk_ptr >= input && extended_blk_ptr < extended_blk_end) {
        value = *extended_blk_ptr;
      }
      temp[shm_pos(pos)] = pre(value);
    }

    // Step 2: Calculate prefix sum of the extended block, in place
    // (note: __syncthreads already happens inside)
    PrefixSumSharedMem(temp, pow2, shm_pos);
    __syncthreads();

    // Step 3: Compute the output, the sum in window, by subtracting two values of the prefix sum
    for (int pos = threadIdx.x; pos < logical_block; pos += blockDim.x) {
      acc_t<In> out_val = pre(logical_block_in_ptr[shm_pos(pos)]) + temp[shm_pos(window + pos)] -
                          temp[shm_pos(pos)];
      logical_block_out_ptr[pos] = ConvertSat<Out>(post(out_val));
    }
  }
}

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
  auto *sample_descs_cpu = ctx.scratchpad->AllocatePinned<SampleDesc>(nsamples);

  int64_t max_len;
  for (int i = 0; i < nsamples; i++) {
    auto &sample = sample_descs_cpu[i];
    sample.out = out[i].data;
    sample.in = in[i].data;
    sample.len = in[i].shape.num_elements();
    max_len = std::max(sample.len, max_len);
  }
  auto *sample_descs_gpu =
      ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

  int window_len = args.window_size;
  int logical_block = needs_reset<InputType>::value && args.reset_interval > 0 ?
                          next_pow2(args.reset_interval) :
                          4 * window_len;
  int pow2 = next_pow2(window_len + logical_block);

  dim3 grid(std::min<int64_t>(1024, div_ceil(max_len, 32)), nsamples, 1);
  int block_sz = 256;
  int64_t shm_sz = 2 * pow2 * sizeof(double);

  square pre;
  divide post(window_len);
  SlidingWindowSum<float, InputType><<<grid, block_sz, shm_sz, ctx.gpu.stream>>>(
      sample_descs_gpu, logical_block, window_len, pow2, pre, post);

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
