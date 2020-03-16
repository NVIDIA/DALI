// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
#define _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH


#include "dali/core/util.h"
#include "dali/kernels/reduce/reduce.h"
#include "dali/kernels/reduce/reduce_all_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_inline.cuh"

namespace dali {
namespace kernels {

/**
 * @brief This function is used for reducing innermost dimension with small extent.
 *
 * The reduction is done in a single pass, with each thread completely reducing
 * the inner dimension.
 * The implementation uses tree reduction up to 256 elements and above that
 * simple sum of 256-element blocks.
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerSmall(Acc *out, const In *in, int64_t n_outer, int n_inner,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_size_x = static_cast<int64_t>(gridDim.x) * blk_size
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_size_x) {
    const float *base = in + outer * n_inner;
    float acc = pp(__ldg(base));

    out[outer] = reductions::ReduceInner(base, n_inner, reduce, pp);
  }
}

/**
 * @brief This kernel is used for reducing innermost dimension with large extent.
 *
 * The reduction needs at least one more level to complete. Output buffer `out`
 * will contain n_outer * gridDim.x elements
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerLarge(Acc *out, const In *in, int64_t n_outer, int64_t n_inner,
                                 int inner_block, int macroblock_size,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  constexpr int64_t blk_size = 32*32;  // block must be warpSize * warpSize for BlockReduce
  constexpr int max_macroblock_size = 32;  // 32 blocks

  const int64_t grid_size = gridDim.x * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  int64_t offset = blockIdx.x * blk_size + flat_tid;
  Acc val = offset < sample_size ? pp(in[offset]) : reduce.template neutral<Acc>();
  for (offset += grid_size; offset < sample_size; offset += grid_size) {
    reduce(val, pp(in[offset]));
  BlockReduce(out, &in[outer * n_inner], [outer]() { return outer; });
}

template <typename Out, typename In>
struct ReduceInnerSampleDesc {
  Out *out;
  const In *in;
  int64_t shape[2];
  int inner_blocks, inner_block_size;
};

template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceInnerKernel(const ReduceInnerSampleDesc<Out, In> *samples,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  int sample = threadIdx.y;
  int64_t n_outer = samples[sample].shape[0];
  int64_t n_inner = samples[sample].shape[1];
  int64_t sample_size = n_outer * n_inner;

  if (n_inner <= 1024 && samples[i].inner_blocks == 1)
    ReduceInnerSmall(&out[sample], in[sample], n_outer, n_inner, reduce, pp);
  else
    ReduceInnerLarge(&out[sample], in[sample], n_outer, n_inner, reduce, pp);
}


}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
