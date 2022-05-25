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

#ifndef DALI_KERNELS_COMMON_FIND_FIND_FIRST_LAST_GPU_CUH_
#define DALI_KERNELS_COMMON_FIND_FIND_FIRST_LAST_GPU_CUH_

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_common.cuh"

namespace dali {
namespace kernels {
namespace find {

namespace {

/**
 * @brief Sample descriptor
 */
template <typename T, typename Predicate>
struct SampleDesc {
  int64_t *a_ptr, *b_ptr;  // represents (first, last), (begin, end), or (begin, length) depending
                           // on the OutputProcessor
  Predicate predicate;
  const T *in;
  int64_t len;
};

/**
 * @brief Represents first, last (or other range representations) coordinates
 */
struct pair_i64 {
    int64_t a = 0;
    int64_t b = 0;
};

/**
 * @brief First and last position
 */
struct first_last {
  DALI_HOST_DEV DALI_FORCEINLINE pair_i64 operator()(pair_i64 x) const noexcept {
    return x;
  }
};

/**
 * @brief Begin (inclusive) and End (exclusive) of the region
 */
struct begin_end {
  DALI_HOST_DEV DALI_FORCEINLINE pair_i64 operator()(pair_i64 x) const noexcept {
    return {x.a, x.b + 1};
  }
};

/**
 * @brief Begin (inclusive) and Length of the region
 */
struct begin_length {
  DALI_HOST_DEV DALI_FORCEINLINE pair_i64 operator()(pair_i64 x) const noexcept {
    return {x.a, x.b - x.a + 1};
  }
};

/**
 * @brief Extract the position of the first and last position that satisfies a predicate
 *
 * @remarks Calculates a double reduction (min, max) in one go
 *
 * @tparam T Input data type
 * @tparam Predicate Predicate that must be satisfied
 * @tparam OutFormat Optional Transformation to first, last coordinates
 * @param samples Sample descriptors, including a per-sample predicate
 * @param format Optional output coordinate transformation
 */
template <typename T, typename Predicate, typename OutFormat = first_last>
__global__ void FindFirstLastImpl(SampleDesc<T, Predicate> *samples, OutFormat format = {}) {
  int sample_idx = blockIdx.y;
  auto &sample = samples[sample_idx];
  const T *input = sample.in;
  int64_t sample_len = sample.len;
  auto &predicate = sample.predicate;

  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  reductions::min first_reduction;
  reductions::max last_reduction;

  int64_t first_neutral = first_reduction.template neutral<int64_t>();
  int64_t last_neutral = last_reduction.template neutral<int64_t>();
  int64_t first = first_neutral;
  int64_t last = last_neutral;

  // similar concept as in reduction kernels

  int64_t idx = blockIdx.x * blk_size + flat_tid;
  for (; idx < sample_len; idx += grid_size) {
    int64_t tmp_idx = predicate(input[idx]) ? idx : -1;
    int64_t first_candidate = tmp_idx < 0 ? first_reduction.template neutral<int64_t>() : tmp_idx;
    int64_t last_candidate = tmp_idx < 0 ? last_reduction.template neutral<int64_t>() : tmp_idx;

    first_reduction(first, first_candidate);
    last_reduction(last, last_candidate);
  }

  BlockReduce(first, first_reduction);
  BlockReduce(last, last_reduction);
  if (flat_tid == 0) {
    if (first == first_neutral || last == last_neutral) {
      *sample.a_ptr = 0;
      *sample.b_ptr = 0;
    } else {
      auto tmp = format(pair_i64{first, last});
      *sample.a_ptr = tmp.a;
      *sample.b_ptr = tmp.b;
    }
  }
}

}  // namespace

/**
 * @brief Extract the position of the first and last position that satisfies a predicate
 */
class FindFirstLastGPU {
 public:
  template <typename T>
  KernelRequirements Setup(KernelContext &context, const InListGPU<T, 1> &in) {
    KernelRequirements req;
    int nsamples = in.size();
    TensorListShape<0> out_sh(nsamples);
    req.output_shapes = {out_sh, out_sh};  // begin, length outputs
    return req;
  }

  template <typename T, typename Predicate, typename OutputFormat = first_last>
  void Run(KernelContext &ctx,
           const OutListGPU<int64_t, 0> &begin,
           const OutListGPU<int64_t, 0> &length,
           const InListGPU<T, 1> &in,
           span<Predicate> predicates = {},
           OutputFormat format = {}) {
    int nsamples = in.shape.num_samples();
    auto *sample_descs_cpu = ctx.scratchpad->AllocatePinned<SampleDesc<T, Predicate>>(nsamples);

    int64_t max_len;
    for (int i = 0; i < nsamples; i++) {
      auto &sample = sample_descs_cpu[i];
      sample.a_ptr = begin[i].data;
      sample.b_ptr = length[i].data;
      sample.in = in[i].data;
      sample.len = in[i].shape.num_elements();
      // Allowing a default or a single predicate for the whole batch
      sample.predicate = predicates.empty()     ? Predicate{} :
                         predicates.size() == 1 ? predicates[0] :
                                                  predicates[i];
      max_len = std::max(sample.len, max_len);
    }
    auto *sample_descs_gpu =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

    dim3 grid(1, nsamples);  // 1 output bin per sample (reduction to scalar)
    dim3 block(32, 32);      // expected by BlockReduce
    FindFirstLastImpl<T, Predicate, OutputFormat>
        <<<grid, block, 0, ctx.gpu.stream>>>(sample_descs_gpu, format);
    CUDA_CALL(cudaGetLastError());
  }
};

}  // namespace find
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_FIND_FIND_FIRST_LAST_GPU_CUH_
