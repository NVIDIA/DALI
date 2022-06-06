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
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/reduce/find_reduce.cuh"
#include "dali/kernels/reduce/reduce_common.cuh"

namespace dali {
namespace kernels {
namespace find_first_last {

namespace {

/**
 * @brief Sample descriptor
 */
template <typename T, typename Idx, typename Predicate, typename Postprocessor = nullptr_t>
struct SampleDesc {
  Idx *a_ptr, *b_ptr;  // represents (first, last), (begin, end), or (begin, length) depending
                       // on Postprocessor
  const Predicate *predicate;
  const Postprocessor *post;
  const T *in;
  int64_t len;
};

template <typename Idx, typename Postprocessor>
struct PostprocessSampleDesc {
  Idx *a_ptr;
  Idx *b_ptr;
  const Postprocessor *post;
};

/**
 * @brief Represents first, last (or other range representations) coordinates
 */
template <typename Idx>
struct idx_pair {
    Idx a = 0;
    Idx b = 0;
};

/**
 * @brief First and last position
 */
template <typename Idx>
struct first_last {
  DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> operator()(idx_pair<Idx> x) const noexcept {
    return x;
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> neutral() const noexcept {
    return {-1, -1};
  }
};

/**
 * @brief Begin (inclusive) and End (exclusive) of the region
 */
template <typename Idx>
struct begin_end {
  DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> operator()(idx_pair<Idx> x) const noexcept {
    return {x.a, x.b + 1};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> neutral() const noexcept {
    return {0, 0};  // empty range
  }
};

/**
 * @brief Begin (inclusive) and Length of the region
 */
template <typename Idx>
struct begin_length {
  DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> operator()(idx_pair<Idx> x) const noexcept {
    return {x.a, x.b - x.a + 1};
  }

  constexpr DALI_HOST_DEV DALI_FORCEINLINE idx_pair<Idx> neutral() const noexcept {
    return {0, 0};  // empty range
  }
};

template <typename Postprocessor, typename Idx>
__device__ void Postprocess(Postprocessor &post, Idx &a, Idx &b, idx_pair<Idx> neutral) {
  auto tmp = post.neutral();
  if (a != neutral.a && b != neutral.b)
    tmp = post({a, b});
  a = tmp.a;
  b = tmp.b;
}


template <typename Idx>
__device__ void Postprocess(nullptr_t &, Idx &, Idx &, idx_pair<Idx>) {}

/**
 * @brief Extract the position of the first and last position that
 * satisfies a predicate, optionally postprocessing the coordinates
 *
 * @remarks Calculates a double reduction (min, max) in one go
 */
template <typename T, typename Idx, typename Predicate, typename Postprocessor = nullptr_t>
__global__ void FindFirstLastImpl(SampleDesc<T, Idx, Predicate, Postprocessor> *samples) {
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
  auto predicate = sample.predicate ? *sample.predicate : Predicate{};
  auto post = sample.post ? *sample.post : Postprocessor{};

  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int64_t flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  reductions::min first_reduction;
  reductions::max last_reduction;

  constexpr Idx first_neutral = first_reduction.template neutral<Idx>();
  constexpr Idx last_neutral = last_reduction.template neutral<Idx>();
  constexpr idx_pair<Idx> neutral{first_neutral, last_neutral};
  Idx first = first_neutral;
  Idx last = last_neutral;

  // similar concept as in reduction kernels
  int64_t idx = blockIdx.x * blk_size + flat_tid;
  for (; idx < sample_len; idx += grid_size) {
    Idx tmp_idx = predicate(input[idx]) ? idx : -1;
    Idx first_candidate = tmp_idx < 0 ? first_neutral : tmp_idx;
    Idx last_candidate = tmp_idx < 0 ? last_neutral : tmp_idx;

    first_reduction(first, first_candidate);
    last_reduction(last, last_candidate);
  }

  BlockReduce(first, first_reduction);
  BlockReduce(last, last_reduction);
  if (flat_tid == 0) {
    Postprocess(post, first, last, neutral);
    *sample.a_ptr = first;
    *sample.b_ptr = last;
  }
}

/**
 * @brief Converts first, last coordinates to a different format
 */
template <typename Idx, typename Postprocessor>
__global__ void PostprocessKernelImpl(PostprocessSampleDesc<Idx, Postprocessor> *samples,
                                      int nsamples, idx_pair<Idx> neutral) {
  for (int idx = threadIdx.x; idx < nsamples; idx += blockDim.x) {
    auto &sample = samples[idx];
    Postprocessor post;
    if (sample.post)
      post = *sample.post;
    Postprocess(post, *sample.a_ptr, *sample.b_ptr, neutral);
  }
}

}  // namespace

/**
 * @brief Extract the position of the first and last position that satisfies a predicate, optionally
 * postprocessing the coordinates
 */
template <typename T, typename Idx, typename Predicate, typename Postprocessor = nullptr_t>
class FindFirstLastGPU {
 public:
  KernelRequirements Setup(KernelContext &context, const InListGPU<T, 1> &in) {
    KernelRequirements req;
    int nsamples = in.size();
    TensorListShape<0> out_sh(nsamples);
    req.output_shapes = {out_sh, out_sh};  // begin, length outputs
    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Idx, 0> &begin,
           const OutListGPU<Idx, 0> &length,
           const InListGPU<T, 1> &in,
           const InListGPU<Predicate, 0> predicates,
           const InListGPU<Postprocessor, 0> postprocessors = {}) {
    int nsamples = in.shape.num_samples();
    // Two different implementations. For larger batches, a specialized one-pass implementation is
    // used that calculates the postprocessed first,last coordinates in a single kernel launch. This
    // implementation uses a block per sample, so it is inefficient for smaller batches. For larger
    // batches, three kernel launches are used:
    // - Find the first element
    // - Find the last element
    // - Postprocess the output coordinates
    if (nsamples >= 16) {
      RunFindFirstLastKernel(ctx, begin, length, in, predicates, postprocessors);
    } else {
      first_.Setup(ctx, in.shape);
      last_.Setup(ctx, in.shape);
      first_.Run(ctx, begin, in, predicates);
      last_.Run(ctx, length, in, predicates);
      if (!std::is_same<Postprocessor, nullptr_t>::value) {
        // that's what FindReduceGPU gives us for "not-found"
        constexpr idx_pair<Idx> neutral{-1, -1};
        RunPostprocessKernel(ctx, begin, length, neutral, postprocessors);
      }
    }
  }

 private:
  void RunFindFirstLastKernel(KernelContext ctx, const OutListGPU<Idx, 0> &first,
                              const OutListGPU<Idx, 0> &last, const InListGPU<T, 1> &in,
                              const InListGPU<Predicate, 0> predicates,
                              const InListGPU<Postprocessor, 0> postprocessors = {}) {
    int nsamples = in.shape.num_samples();
    auto *sample_descs_cpu =
        ctx.scratchpad->AllocatePinned<SampleDesc<T, Idx, Predicate, Postprocessor>>(nsamples);

    for (int i = 0; i < nsamples; i++) {
      auto &sample = sample_descs_cpu[i];
      sample.a_ptr = first[i].data;
      sample.b_ptr = last[i].data;
      sample.in = in[i].data;
      sample.len = in[i].shape.num_elements();
      sample.predicate = predicates[i].data;
      if (!postprocessors.empty())
        sample.post = postprocessors[i].data;
    }
    auto *sample_descs_gpu =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(sample_descs_cpu, nsamples));

    dim3 grid(1, nsamples);  // 1 output bin per sample (reduction to scalar)
    dim3 block(32, 32);      // expected by BlockReduce

    FindFirstLastImpl<T, Idx, Predicate, Postprocessor>
        <<<grid, block, 0, ctx.gpu.stream>>>(sample_descs_gpu);
    CUDA_CALL(cudaGetLastError());
  }

  void RunPostprocessKernel(KernelContext ctx, const OutListGPU<Idx, 0> &first,
                            const OutListGPU<Idx, 0> &last,
                            idx_pair<Idx> neutral,
                            const InListGPU<Postprocessor, 0> postprocessors = {}) {
    int nsamples = first.num_samples();
    int block = nsamples > 256 ? 256 : nsamples;

    assert(postprocessors.size() == nsamples || postprocessors.empty());
    auto *postprocess_samples_cpu =
        ctx.scratchpad->AllocatePinned<PostprocessSampleDesc<Idx, Postprocessor>>(nsamples);
    for (int i = 0; i < nsamples; i++) {
      auto &sample = postprocess_samples_cpu[i];
      sample.a_ptr = first[i].data;
      sample.b_ptr = last[i].data;
      sample.post = postprocessors.empty() ? nullptr : postprocessors[i].data;
    }
    auto *postprocess_samples_gpu =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(postprocess_samples_cpu, nsamples));

    PostprocessKernelImpl<Idx, Postprocessor>
        <<<1, block, 0, ctx.gpu.stream>>>(postprocess_samples_gpu, nsamples, neutral);
    CUDA_CALL(cudaGetLastError());
  }

  FindReduceGPU<Idx, T, Predicate, reductions::min> first_;
  FindReduceGPU<Idx, T, Predicate, reductions::max> last_;
};

}  // namespace find_first_last
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_FIND_FIND_FIRST_LAST_GPU_CUH_
