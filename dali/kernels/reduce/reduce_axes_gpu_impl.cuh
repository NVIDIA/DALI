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
  const int64_t grid_size_x = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_size_x) {
    const float *base = in + outer * n_inner;
    out[outer] = reductions::ThreadReduce<Acc>(base, n_inner, 1, reduce, pp);
  }
}

/**
 * @brief This kernel is used for reducing innermost dimension with large extent.
 *
 * The reduction needs at least one more level to complete. Output buffer
 * will contain n_outer * num_macroblocks elements.
 *
 * After this function is used, another level of reduction may be necessary,
 * depending on the value num_macroblocks.
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerLarge(Acc *out, const In *in, int64_t n_outer, int64_t n_inner,
                                 int num_macroblocks, int macroblock_size,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  constexpr int64_t blk_size = 32*32;  // block must be warpSize * warpSize for BlockReduce

  const int64_t total_blocks = n_outer * num_macroblocks;

  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  int outer_shift = __ffs(num_macroblocks) - 1;
  int inner_mask = num_macroblocks - 1;

  for (int64_t idx = blockIdx.x; idx < total_blocks; idx += gridDim.x) {
    int64_t outer = idx >> outer_shift;
    int64_t inner_macroblock = idx & inner_mask;
    int64_t inner_start = inner_macroblock * macroblock_size;
    int64_t inner_end = min(n_inner, inner_start + macroblock_size);

    const In *base = &in[outer * n_inner];

    Acc val = reduce.template neutral<Acc>();

    // reduce macroblock to a block - each thread reduces its own block-strided slice
    bool first = true;
    for (int64_t inner = inner_start + flat_tid; inner < inner_end; inner += blk_size) {
      auto x = pp(__ldg(base + inner));
      if (first) {
        val = x;
        first = false;
      } else {
        reduce(val, x);
      }
    }

    if (idx != blockIdx.x)  // not needed in first iteration:
      __syncthreads();      // make sure that the shared memory used by BlockReduce is ready

    BlockReduce(out, val, reduce, [idx]() { return idx; });
  }
}

template <typename Out, typename In>
struct ReduceSampleDesc {
  Out *out;
  const In *in;
  int64_t n_outer;    // volume of the outermost (non-reduced) dimensions
  int64_t n_reduced;  // volume of the reduced
  int64_t n_inner;    // volume of the innermost (non-reduced) dimensions

  /**  number of macroblocks in reduced dimension - must be a power of 2 for easier division */
  int num_macroblocks;

  /**
   * @brief Size, in elements, of the macroblock in reduced dimension - does *not* need to be
   * aligned on block boundary.
   */
  int macroblock_size;
};


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInner(const ReduceSampleDesc<Acc, In> &sample,
                            Reduction reduce = {}, Preprocess pp = {}) {
  assert(sample.n_inner == 1);
  int64_t n_outer = sample.n_outer;
  int64_t n_reduced = sample.n_reduced;
  Acc *out = sample.out;
  const In *in = sample.in;

  // TODO(michalz) the 32-256 reduction is very slow - we need a dedicated function for this range
  if (n_reduced < 64 && sample.num_macroblocks == 1) {
    ReduceInnerSmall(out, in, n_outer, n_reduced, reduce, pp);
  } else {
    ReduceInnerLarge(out, in, n_outer, n_reduced,
                     sample.num_macroblocks, sample.macroblock_size, reduce, pp);
  }
}

template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceInnerKernel(const ReduceSampleDesc<Acc, In> *samples,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  ReduceInner(samples[blockIdx.y], reduce, pp);
}

// Reduction over other dimensions (non-innermost)

template <typename Div, typename Mod, typename X, typename Y>
DALI_HOST_DEV DALI_FORCEINLINE void _divmod(Div &div, Mod &mod, X x, Y y) {
  div = x/y;
  mod = x%y;
}

template <typename Div, typename Mod, typename X, typename Y>
DALI_HOST_DEV DALI_FORCEINLINE void divmod(Div &div, Mod &mod, X x, Y y) {
  switch (y) {
    case 1:
      div = x; mod = 0; break;
    case 2:
      _divmod(div, mod, x, 2); break;
    case 3:
      _divmod(div, mod, x, 3); break;
    case 4:
      _divmod(div, mod, x, 4); break;
    case 5:
      _divmod(div, mod, x, 5); break;
    case 6:
      _divmod(div, mod, x, 6); break;
    case 7:
      _divmod(div, mod, x, 7); break;
    case 8:
      _divmod(div, mod, x, 8); break;
    case 9:
      _divmod(div, mod, x, 9); break;
    case 10:
      _divmod(div, mod, x, 10); break;
    case 11:
      _divmod(div, mod, x, 11); break;
    case 12:
      _divmod(div, mod, x, 12); break;
    case 13:
      _divmod(div, mod, x, 13); break;
    case 14:
      _divmod(div, mod, x, 14); break;
    case 15:
      _divmod(div, mod, x, 15); break;
    case 16:
      _divmod(div, mod, x, 16); break;
    default:
      if ((y & (y-1)) == 0) {
        div = x >> (__ffs(y) - 1);
        mod = x & (y-1);
      } else {
        _divmod(div, mod, x, y); break;
      }
  }
}


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceOtherSmall(const ReduceSampleDesc<Acc, In> &sample,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  int64_t n_outer = sample.n_outer;
  int n_reduced = sample.n_reduced;
  int64_t n_inner = sample.n_inner;
  int64_t n_non_reduced = n_inner * n_outer;
  int64_t outer_stride = n_inner * n_reduced;

  Acc *out = sample.out;
  const In *in = sample.in;

  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_size_x = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t idx = base_idx; idx < n_non_reduced; idx += grid_size_x) {
    int64_t outer, inner;
    divmod(outer, inner, idx, n_inner);
    const float *base = in + outer * outer_stride + inner;
    out[idx] = reductions::ThreadReduce<Acc>(base, n_reduced, n_inner, reduce, pp);
  }
}


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceOtherLarge(const ReduceSampleDesc<Acc, In> &sample,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  /*int64_t n_outer = sample.n_outer;
  int n_reduced = sample.n_reduced;
  int64_t n_inner = sample.n_inner;
  int64_t n_non_reduced = n_inner * n_outer;
  int64_t outer_stride = n_inner * n_reduced;

  Acc *out = sample.out;
  const In *in = sample.in;

  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_size_x = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t idx = base_idx; idx < n_non_reduced; idx += grid_size_x) {
    int64_t outer, inner;
    divmod(outer, inner, idx, n_inner);
    const float *base = in + outer * outer_stride + inner;
    out[outer] = reductions::ThreadReduce<Acc>(base, n_reduced, n_inner, reduce, pp);
  }*/
  assert(!"not implemented");
}


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceOtherKernel(const ReduceSampleDesc<Acc, In> *samples,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  auto sample = samples[blockIdx.y];
  int64_t n_inner = sample.n_inner;
  if (n_inner == 1) {
    ReduceInner(sample, reduce, pp);
    return;
  }

  if (sample.n_reduced < 64) {
    ReduceOtherSmall(sample, reduce, pp);
  } else {
    ReduceOtherLarge(sample, reduce, pp);
  }

}

}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
