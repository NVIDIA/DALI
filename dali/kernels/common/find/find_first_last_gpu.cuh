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
namespace find_first_last {

namespace {

/**
 * @brief Sample descriptor
 */
template <typename T, typename Idx, typename Predicate, typename OutFormat>
struct SampleDesc {
  Idx *a_ptr, *b_ptr;  // represents (first, last), (begin, end), or (begin, length) depending
                       // on OutFormat
  Predicate predicate;
  OutFormat format;
  const T *in;
  int64_t len;
};

/**
 * @brief Represents first, last (or other range representations) coordinates
 */
template <typename Idx>
struct pair_idx {
    Idx a = 0;
    Idx b = 0;
};

/**
 * @brief First and last position
 */
template <typename Idx>
struct first_last {
  DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> operator()(pair_idx<Idx> x) const noexcept {
    return x;
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> neutral() const noexcept {
    return {-1, -1};
  }
};

/**
 * @brief Begin (inclusive) and End (exclusive) of the region
 */
template <typename Idx>
struct begin_end {
  DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> operator()(pair_idx<Idx> x) const noexcept {
    return {x.a, x.b + 1};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> neutral() const noexcept {
    return {0, 0};  // empty range
  }
};

/**
 * @brief Begin (inclusive) and Length of the region
 */
template <typename Idx>
struct begin_length {
  DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> operator()(pair_idx<Idx> x) const noexcept {
    return {x.a, x.b - x.a + 1};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE pair_idx<Idx> neutral() const noexcept {
    return {0, 0};  // empty range
  }
};

/**
 * @brief Extract the position of the first and last position (or a derived representation) that
 * satisfies a predicate
 *
 *
 * @remarks Calculates a double reduction (min, max) in one go
 *
 * @tparam T Input data type
 * @tparam Idx Index data type (int64_t, int32_t)
 * @tparam Predicate Predicate that must be satisfied
 * @tparam OutFormat Optional Transformation to first, last coordinates
 * @param samples Sample descriptors, including a per-sample predicate
 * @param format Optional output coordinate transformation
 */
template <typename T, typename Idx, typename Predicate, typename OutFormat>
__global__ void FindFirstLastImpl(SampleDesc<T, Idx, Predicate, OutFormat> *samples) {
  // This kernel is an adapted version of other reduce kernels (see ReduceAllBatchedKernel)
  // This kernel processes 1024-element blocks laid out as 32x32.
  // Grid is flat 2D and blockIdx.x corresponds to an output bin (a single element in this case)
  // and blockIdx.y corresponds to sample in the batch.
  // First, each thread goes with x-grid-sized stride over the data and iteratively reduces
  // in a local variable.
  // Then the local variable is reduced over the block.
  int sample_idx = blockIdx.y;
  auto &sample = samples[sample_idx];
  const T *input = sample.in;
  int64_t sample_len = sample.len;
  auto &predicate = sample.predicate;
  auto &format = sample.format;

  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int64_t flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  reductions::min first_reduction;
  reductions::max last_reduction;

  Idx first_neutral = first_reduction.template neutral<Idx>();
  Idx last_neutral = last_reduction.template neutral<Idx>();
  Idx first = first_neutral;
  Idx last = last_neutral;

  // similar concept as in reduction kernels

  int64_t idx = blockIdx.x * blk_size + flat_tid;
  for (; idx < sample_len; idx += grid_size) {
    Idx tmp_idx = predicate(input[idx]) ? idx : -1;
    Idx first_candidate = tmp_idx < 0 ? first_reduction.template neutral<Idx>() : tmp_idx;
    Idx last_candidate = tmp_idx < 0 ? last_reduction.template neutral<Idx>() : tmp_idx;

    first_reduction(first, first_candidate);
    last_reduction(last, last_candidate);
  }

  BlockReduce(first, first_reduction);
  BlockReduce(last, last_reduction);
  if (flat_tid == 0) {
    auto tmp =
        (first == first_neutral || last == last_neutral) ? format.neutral() : format({first, last});
    *sample.a_ptr = tmp.a;
    *sample.b_ptr = tmp.b;
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

  /**
   * @brief Run from sample descriptors directly in GPU memory
   * @remarks Convenient if sample descriptors are to be generated on GPU directly
   */
  template <typename T, typename Idx, typename Predicate, typename OutFormat>
  void Run(KernelContext &ctx, SampleDesc<T, Idx, Predicate, OutFormat> *sample_descs_gpu,
           int nsamples) {
    dim3 grid(1, nsamples);  // 1 output bin per sample (reduction to scalar)
    dim3 block(32, 32);      // expected by BlockReduce
    FindFirstLastImpl<T, Idx, Predicate, OutFormat>
        <<<grid, block, 0, ctx.gpu.stream>>>(sample_descs_gpu);
    CUDA_CALL(cudaGetLastError());
  }

  /**
   * @brief Run from outputs, inputs, and predicates
   * @remarks Convenient when data is populated on the host
   */
  template <typename T, typename Idx, typename Predicate, typename OutFormat>
  void Run(KernelContext &ctx,
           const OutListGPU<Idx, 0> &begin,
           const OutListGPU<Idx, 0> &length,
           const InListGPU<T, 1> &in,
           span<Predicate> predicates = {},
           span<OutFormat> formaters = {}) {
    int nsamples = in.shape.num_samples();
    auto *sample_descs_cpu =
        ctx.scratchpad->AllocatePinned<SampleDesc<T, Idx, Predicate, OutFormat>>(nsamples);

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

      // Allowing a default or a single output formater for the whole batch
      sample.format = formaters.empty()     ? OutFormat{} :
                      formaters.size() == 1 ? formaters[0] :
                                              formaters[i];
    }
    auto *sample_descs_gpu =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

    Run<T, Idx, Predicate, OutFormat>(ctx, sample_descs_gpu, nsamples);
  }
};

}  // namespace find_first_last
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_FIND_FIND_FIRST_LAST_GPU_CUH_
