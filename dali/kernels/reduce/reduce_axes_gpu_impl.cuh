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
#include "dali/kernels/reduce/reduce_common.cuh"
#include "dali/kernels/reduce/online_reducer.h"

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
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_stride) {
    const float *base = in + outer * n_inner;
    out[outer] = reductions::ThreadReduce<Acc>(base, n_inner, 1, reduce, pp);
  }
}

/**
 * @brief This function is used for reducing innermost dimension with medium extent.
 *
 * The reduction is done by a warp.
 * After sequential step, warp reduction is performed.
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerMedium(Acc *out, const In *in, int64_t n_outer, int n_inner,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.y;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_stride) {
    const float *base = in + outer * n_inner + threadIdx.x;
    int n = (n_inner + 31 - threadIdx.x) >> 5;
    Acc v = reductions::ThreadReduce<Acc>(base, n, 32, reduce, pp);
    WarpReduce(v, reduce);
    if (threadIdx.x == 0) {
      out[outer] = v;
    }
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
  const int blk_size = 32*blockDim.y;  // block must be warpSize * warpSize for BlockReduce

  const int total_blocks = n_outer * num_macroblocks;

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

    if (BlockReduce(val, reduce))
      out[idx] = val;
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
  int64_t n_outer = sample.n_outer;
  int64_t n_reduced = sample.n_reduced;
  Acc *out = sample.out;
  const In *in = sample.in;

  if (n_reduced < 32 && sample.num_macroblocks == 1) {
    ReduceInnerSmall(out, in, n_outer, n_reduced, reduce, pp);
  } else if (n_reduced < 1024 && sample.num_macroblocks == 1) {
    ReduceInnerMedium(out, in, n_outer, n_reduced, reduce, pp);
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

/**
 * Each *thread* performs independent, full reduction
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__
void ReduceMiddleSmall(const ReduceSampleDesc<Acc, In> &sample,
                           Reduction reduce = {}, Preprocess pp = {}) {
  int64_t n_outer = sample.n_outer;
  int n_reduced = sample.n_reduced;
  int64_t n_inner = sample.n_inner;
  int64_t n_non_reduced = n_inner * n_outer;
  int64_t outer_stride = n_inner * n_reduced;

  Acc *out = sample.out;
  const In *in = sample.in;
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t idx = base_idx; idx < n_non_reduced; idx += grid_stride) {
    int64_t outer, inner;
    outer = idx / n_inner;
    inner = idx % n_inner;
    const float *base = in + outer * outer_stride + inner;
    out[idx] = reductions::ThreadReduce<Acc>(base, n_reduced, n_inner, reduce, pp);
  }
}


/**
 * Each *warp* performs full reduction - blockDim.y is used as a stride in non-reduced dims
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceMiddleMedium(const ReduceSampleDesc<Acc, In> &sample,
                        Reduction reduce = {}, Preprocess pp = {}) {
  int64_t n_outer = sample.n_outer;
  int n_reduced = sample.n_reduced;
  int64_t n_inner = sample.n_inner;
  int64_t n_non_reduced = n_inner * n_outer;
  int64_t outer_stride = n_inner * n_reduced;

  Acc *out = sample.out;
  const In *in = sample.in;

  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.y;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  // we can't add thread_offset to base index, because we need warp convergence at all times
  // to perform warp-sync reduction
  int tid = threadIdx.x;
  int64_t thread_offset = tid * n_inner;
  for (int64_t idx = base_idx; idx < n_non_reduced; idx += grid_stride) {
    int64_t outer, inner;
    outer = idx / n_inner;
    inner = idx % n_inner;
    const float *base = in + outer * outer_stride + thread_offset + inner;

    int n = (n_reduced + 31 - tid) >> 5;
    Acc val = reductions::ThreadReduce<Acc>(base, n, n_inner * 32, reduce, pp);

    WarpReduce(val, reduce);
    if (tid == 0)
      out[idx] = val;
  }
}

namespace reduce_shared {
  extern __shared__ uint8_t shared_tmp[];
}  // namespace reduce_shared

// Works with numerators up to 2^22 and denominators up to 2^20 (tested)
__device__ DALI_FORCEINLINE
int fast_div(int x, int y) {
  return __float2int_rd(x * __frcp_ru(y));
}

// Works with numerators up to 2^22 and denominators up to 2^20 (tested)
__device__ DALI_FORCEINLINE
int fast_mod(int x, int y) {
  return x - fast_div(x, y) * y;
}

/**
 * @brief Reduces the input by fetching contiguous warps of data per block.
 *
 * Each block processes a whole macroblock. Each threads accumulates values
 * assigned to a single output bin in the inner dimension.
 *
 * This function CANNOT work for inner extensts >32.
 * The macroblock size in reduced dimension is limited - see OnlineReducer for limits.
 */
template <typename Acc, typename In, typename Reduction, typename Preprocess = identity>
__device__ void ReduceMiddleLargeInnerSmall(
      const ReduceSampleDesc<Acc, In> &sample,
      Reduction r = {}, Preprocess pp = {}) {
  Acc (*shared_tmp)[33] = reinterpret_cast<Acc (*)[33]>(reduce_shared::shared_tmp);

  Acc *out = sample.out;
  const In *in = sample.in;

  const int n_inner = sample.n_inner;
  const int num_macroblocks = sample.num_macroblocks;
  const int macroblock_size = sample.macroblock_size;
  const int64_t macroblock_stride = static_cast<int64_t>(macroblock_size) * n_inner;

  const int64_t total_macroblocks = sample.n_outer * num_macroblocks;

  int outer_shift = __ffs(num_macroblocks) - 1;
  int mblock_mask = num_macroblocks - 1;

  const int warps = blockDim.y;  // (blockDim.x * blockDim.y + 31) >> 5;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int blk_size = blockDim.x * blockDim.y;

  const int64_t outer_stride = n_inner * sample.n_reduced;

  const int block_mod = blk_size % n_inner;
  int lane_offset0 = -((warp * 32) % n_inner);
  if (lane_offset0 < 0)
    lane_offset0 += n_inner;

  for (int w = warp; w < 32; w += warps)
    shared_tmp[w][lane] = r.template neutral<Acc>();

  for (int64_t macroblock = blockIdx.x; macroblock < total_macroblocks; macroblock += gridDim.x) {
    __syncthreads();
    int64_t outer_ofs = (macroblock >> outer_shift) * outer_stride;
    int mblock = macroblock & mblock_mask;
    int64_t macroblock_start = mblock * macroblock_stride;
    int64_t macroblock_end = min(macroblock_start + macroblock_stride, outer_stride);
    OnlineReducer<Acc, Reduction, 256, 16> red;
    red.reset(r);
    int lane_offset = lane_offset0;
    for (int64_t i = macroblock_start; i < macroblock_end; i += blk_size) {
      int64_t idx = i + tid;
      int64_t idx0 = idx + lane_offset;

      Acc v = lane + lane_offset < 32 &&
              idx0 < macroblock_end
                ? pp(__ldg(in + outer_ofs + idx0))
                : r.template neutral<Acc>();
      if (lane + lane_offset >= n_inner && lane < n_inner) {
        int64_t idx1 = idx0 - n_inner;
        if (idx1 < macroblock_end)
          r(v, pp(__ldg(in + outer_ofs + idx1)));
      }
      red.add(v, r);

      lane_offset -= block_mod;
      if (lane_offset < 0)
        lane_offset += n_inner;
    }
    Acc v = red.result(r);

    shared_tmp[warp][lane] = v;
    __syncthreads();
    for (int inner = warp; inner < n_inner; inner += warps) {
      v = shared_tmp[lane][inner];
      for (int j = inner + n_inner; j < 32; j += n_inner) {
        r(v, shared_tmp[lane][j]);
      }
      WarpReduce(v, r);
      if (lane == 0) {
        out[macroblock * n_inner + inner] = v;
      }
    }
  }
}


/**
 * @brief Reduces the input in tiles shaped warp_size x macroblock_size.
 *
 * Each block processes a tile warp_size x macroblock_size. Then it jumps
 * by grid stride until data is exhausted.
 *
 * This function is not indented for use with small inner extent.
 * The macroblock size in reduced dimension is limited - see OnlineReducer for limits.
 */
template <typename Acc, typename In, typename Reduction, typename Preprocess = identity>
__device__ void ReduceMiddleLargeInnerMedium(
      const ReduceSampleDesc<Acc, In> &sample,
      Reduction r = {}, Preprocess pp = {}) {
  Acc (*shared_tmp)[33] = reinterpret_cast<Acc (*)[33]>(reduce_shared::shared_tmp);

  Acc *out = sample.out;
  const In *in = sample.in;

  const int n_inner = sample.n_inner;
  const int64_t n_reduced = sample.n_reduced;
  const int num_macroblocks = sample.num_macroblocks;
  const int macroblock_size = sample.macroblock_size;

  int outer_shift = __ffs(num_macroblocks) - 1;
  int mblock_mask = num_macroblocks - 1;

  // Start of inner dimension is warp-aligned.
  // We need to know how many warps are required to span entire inner extent.
  int horz_warps = (n_inner + 31) / 32;

  // Caution: fast approximate division ahead!
  //
  // Rationale: grid size is going to be relatively small (up to a few thousand)
  // and number of horizontal warps is also fairly limited.
  // The approximation has been exhaustively tested with numerators up to 2^22 and
  // denominators up to 2^20 and the result is exact.
  // The values expected here are much smaller.
  float rcp_horz_warps = __frcp_ru(horz_warps);  // round up - otherwise it's no good for
                                                 // approximating _integer_ division

  const int total_out_blocks = sample.n_outer * num_macroblocks * horz_warps;

  const int warps = (blockDim.x * blockDim.y + 31) >> 5;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t outer_stride = n_inner * sample.n_reduced;

  for (int w = warp; w < 32; w += warps)
    shared_tmp[w][lane] = r.template neutral<Acc>();

  for (int out_block = blockIdx.x; out_block < total_out_blocks; out_block += gridDim.x) {
    __syncthreads();

    // Caution: fast approximate division ahead!
    int macroblock = __float2int_rd(out_block * rcp_horz_warps);  // integer division - round down
    int horz_warp = out_block - macroblock * horz_warps;  // modulo

    int64_t outer_ofs = (macroblock >> outer_shift) * outer_stride;
    int mblock = macroblock & mblock_mask;

    int64_t macroblock_start = mblock * macroblock_size;
    int64_t macroblock_end = min(macroblock_start + macroblock_size, n_reduced);
    OnlineReducer<Acc, Reduction, 256, 24> red;
    red.reset(r);
    int inner_base = horz_warp * 32;
    int64_t base_idx = outer_ofs + inner_base + lane;
    Acc v;
    if (lane + inner_base < n_inner) {
      for (int64_t i = macroblock_start + warp; i < macroblock_end; i += warps) {
        int64_t offset = base_idx + i * n_inner;
        red.add(pp(__ldg(in + offset)), r);
      }
      v = red.result(r);
    } else {
      v = r.template neutral<Acc>();
    }

    shared_tmp[warp][lane] = v;
    __syncthreads();
    for (int i = warp; i < 32 && i + inner_base < n_inner; i += warps) {
      v = shared_tmp[lane][i];
      WarpReduce(v, r);
      if (lane == 0) {
        int inner = i + inner_base;
        out[macroblock * n_inner + inner] = v;
      }
    }
  }
}


template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceMiddleKernel(const ReduceSampleDesc<Acc, In> *samples,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  auto sample = samples[blockIdx.y];

  if (sample.n_reduced < 1024) {
    ReduceMiddleSmall(sample, reduce, pp);
  } else if (sample.n_inner < 32) {
    ReduceMiddleLargeInnerSmall(sample, reduce, pp);
  } else {
    ReduceMiddleLargeInnerMedium(sample, reduce, pp);
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
