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

#ifndef _DALI_KERNELS_REDUCE_REDUCE_ALL_GPU_IMPL_CUH
#define _DALI_KERNELS_REDUCE_REDUCE_ALL_GPU_IMPL_CUH


#include "dali/core/util.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_common.cuh"

namespace dali {
namespace kernels {

template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceAllKernel(Acc *out, const In *in, int64_t n,
                                Reduction reduce = {}, Preprocess pp = {}) {
  // This kernel processes 1024-element blocks laid out as 32x32.
  // Grid is flat 1D and blockIdx.x corresponds to an output bin.
  // First, each thread goes with grid-sized stride over the data and iteratively reduces
  // in a local variable.
  // Then the local variable is reduced over the block.
  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t idx = blockIdx.x * blk_size + flat_tid;
  Acc val = idx < n ? pp(in[idx]) : reduce.template neutral<Acc>();
  for (idx += grid_size; idx < n; idx += grid_size) {
    reduce(val, pp(in[idx]));
  }
  if (BlockReduce(val, reduce))
    out[blockIdx.x] = val;
}


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceAllBatchedKernel(Acc *out, const In *const *in, const int64_t *in_sizes,
                                       Reduction reduce = {}, const Preprocess *pp = nullptr) {
  // This kernel processes 1024-element blocks laid out as 32x32.
  // Grid is flat 2D and blockIdx.x corresponds to an output bin and blockIdx.y corresponds to
  // sample in the batch.
  // First, each thread goes with x-grid-sized stride over the data and iteratively reduces
  // in a local variable.
  // Then the local variable is reduced over the block.
  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int sample = blockIdx.y;
  Preprocess preprocess = pp ? pp[sample] : Preprocess();
  const int64_t n = in_sizes[sample];
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t idx = blockIdx.x * blk_size + flat_tid;
  Acc val = idx < n ? preprocess(in[sample][idx]) : reduce.template neutral<Acc>();
  for (idx += grid_size; idx < n; idx += grid_size) {
    reduce(val, preprocess(in[sample][idx]));
  }
  if (BlockReduce(val, reduce))
    out[blockIdx.x + blockIdx.y * gridDim.x] = val;
}

/**
 * @brief Reduce evenly-spaced, contiguous blocks in `in` and store blockwise results in `out`.
 *
 * This kernel treats `in` as a group of contiguously stored, but independent, chunks of data,
 * each of `sample_size` elements.
 * The partial result for each chunk is stored in out.
 *
 * `blockDim = 32, 32` (fixed!)
 * `gridDim = (outputs_per_block, number_of_blocks)`
 *
 * @param out         the result
 * @param in          input data, in blocks `sample_size` elements each
 * @param sample_size number of elements in each block
 * @param reduce      the reduction functor
 * @param pp          preprocessing to be applied to each value fetched from `in` before reducing
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceAllBlockwiseKernel(Acc *out, const In *in, int64_t sample_size,
                                         Reduction reduce = {}, const Preprocess *pp = nullptr) {
  // This reduces blocks of size sample_size independently
  const int sample = blockIdx.y;
  Preprocess preprocess = pp ? pp[sample] : Preprocess();
  in += sample * sample_size;  // calculate the base address of this block
  const int64_t blk_size = blockDim.x * blockDim.y;
  const int64_t grid_size = gridDim.x * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t offset = blockIdx.x * blk_size + flat_tid;
  Acc val = offset < sample_size ? preprocess(in[offset]) : reduce.template neutral<Acc>();
  for (offset += grid_size; offset < sample_size; offset += grid_size) {
    reduce(val, preprocess(in[offset]));
  }
  if (BlockReduce(val, reduce))
    out[blockIdx.x + blockIdx.y * gridDim.x] = val;
}


}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_ALL_GPU_IMPL_CUH
