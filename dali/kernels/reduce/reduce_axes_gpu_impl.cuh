// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH_
#define DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH_


#include "dali/core/util.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_all_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_common.cuh"
#include "dali/kernels/reduce/online_reducer.h"
#include "dali/kernels/reduce/reduce_drop_dims.h"

namespace dali {
namespace kernels {

namespace reduce_impl {
/**
 *  A preprocessor bank should satisfy the concept:
 *
 *  ```
 *  struct PreprocessorBank {
 *      SingleValuePreprocessor Get(const i64vec<non_reduced_dims> &pos) const;
 *  };
 *  ```
 *  where SingleValuePreprocessor is a unary functor.
 *
 * non_recuced_dims is either 1 or 2; if it's 1, the dimension 0 is outer (wrt reduced dime)
 * and dimesnion 1 is inner.
 * If non_reduced_dims is 1, then `pos` is the outer dimension.
 */

/**
 * @brief A no-op preprocessor bank,
 */
template <int non_reduced_ndim>
struct IdentityPreprocessor {
  DALI_HOST_DEV DALI_FORCEINLINE
  dali::identity Get(const i64vec<non_reduced_ndim> &) const {
    return {};
  }
};

/**
 * @brief A position-independent bank, wrapping a functor
 */
template <int non_reduced_ndim, typename Functor>
struct UniformPreprocessorBank {
  DALI_HOST_DEV DALI_FORCEINLINE
  Functor Get(const i64vec<non_reduced_ndim> &) const {
    return {};
  }
};

}  // namespace reduce_impl


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

/**
 * @brief This function is used when the reduction is no-op (reduced extent is 1)
 *
 * This function will apply preprocessing and postprocessing, but will not do
 * any actual reduction.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Out, typename In, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceFlatNone(Out *out, const In *in, int64_t n,
                               PreprocessorBank pre_bank, Postprocessor post) {
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t index = base_idx; index < n; index += grid_stride) {
    auto pre = pre_bank.Get({index});
    out[index] = ConvertSat<Out>(post(pre(in[index])));
  }
}


/**
 * @brief This function is used when reducting a degenerate dimension
 *
 * This function stores and optionally postprocesses the reduction's default value.
 *
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename Reduction, typename Postprocessor>
__device__ void ReduceFlatDegenerate(Out *out, int64_t n,
                                     Reduction, Postprocessor post) {
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t index = base_idx; index < n; index += grid_stride) {
    out[index] = ConvertSat<Out>(post(Reduction::template neutral<Acc>()));
  }
}

/**
 * @brief This function is used when the reduction is no-op (reduced extent is 1),
 *        but the reduced axis is not inner, so we need to account for complex addressing
 *        of the preprocessor.
 *
 * This function will apply preprocessing and postprocessing, but will not do
 * any actual reduction.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Out, typename In, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceMiddleNone(Out *out, const In *in, int64_t n_outer, int64_t n_inner,
                                 PreprocessorBank pre_bank, Postprocessor post) {
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  int64_t n = n_inner * n_outer;
  for (int64_t index = base_idx; index < n; index += grid_stride) {
    int64_t outer = index / n_inner;
    int64_t inner = index % n_inner;
    auto pre = pre_bank.Get(i64vec2(outer, inner));
    out[index] = ConvertSat<Out>(post(pre(in[index])));
  }
}

/**
 * @brief This kernel is used when the reduction is no-op (reduced extent is 1)
 *
 * This function will apply preprocessing and postprocessing, but will not do
 * any actual reduction.
 * When the data batch contains only samples where the reduced extent happens to be 1,
 * this kernel will be used.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Out, typename In,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceNoneRawKernel(Out *const *out, const In *const *in, const int64_t *lengths,
                                    PreprocessorBank *pre = nullptr,
                                    Postprocessor *post = nullptr) {
  int sample_idx = blockIdx.y;
  PreprocessorBank pre_bank = pre ? pre[sample_idx] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[sample_idx] : Postprocessor();
  ReduceFlatNone(out[sample_idx], in[sample_idx], lengths[sample_idx], pre_bank, postprocessor);
}


/**
 * @brief This kernel is used when the reduction is no-op (reduced extent is 1)
 *
 * This function will apply preprocessing and postprocessing, but will not do
 * any actual reduction.
 * When the data sub-batch contains only samples where the reduced extent happens to be 1,
 * this kernel will be used.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceFlatNoneKernel(const ReduceSampleDesc<Out, In> *samples,
                                     Reduction reduce = {}, const PreprocessorBank *pre = nullptr,
                                     const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];

  int64_t n = sample.n_outer * sample.n_inner;  // n_reduced is 1, so we don't need to multiply
  Out *out = sample.out;
  const In *in = sample.in;

  ReduceFlatNone(out, in, n, pre_bank, postprocessor);
}

/**
 * @brief This kernel is used when the reduction is a no-op, but the reduced axis is not the inner
 *        one, so complex preprocessing is possible.
 *
 * This function will apply preprocessing and postprocessing, but will not do
 * any actual reduction.
 * When the data sub-batch contains only samples where the reduced extent happens to be 1,
 * this kernel will be used.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<2>,
          typename Postprocessor = identity>
__global__ void ReduceMiddleNoneKernel(const ReduceSampleDesc<Out, In> *samples,
                                       Reduction reduce = {}, const PreprocessorBank *pre = nullptr,
                                       const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];

  Out *out = sample.out;
  const In *in = sample.in;

  if (std::is_same<PreprocessorBank, reduce_impl::IdentityPreprocessor<2>>::value) {
    int64_t n = sample.n_outer * sample.n_inner;  // n_reduced is 1, so we don't need to multiply
    ReduceFlatNone(out, in, n, pre_bank, postprocessor);
  } else {
    ReduceMiddleNone(out, in, sample.n_outer, sample.n_inner, pre_bank, postprocessor);
  }
}


/**
 * @brief This kernel is used when reducing a degenerate middle dimension.
 *
 * This function stores and optionally postprocesses the reduction's default value.
 *
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename Postprocessor = identity>
__global__ void ReduceMiddleDegenerateKernel(const ReduceSampleDesc<Out, In> *samples,
                                             Reduction r,
                                             const Postprocessor *post = nullptr) {
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];
  Out *out = sample.out;

  int64_t n = sample.n_outer * sample.n_inner;
  ReduceFlatDegenerate<Acc>(out, n, r, postprocessor);
}



/**
 * @brief This kernel is used for reducing respective elements of tensors in across samples
 *
 * Grid and block are 1D and are used for indexing the samples; samples themselves are
 * iterated over by threads;
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output coordinate per input sample
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename In,
          typename Reduction,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceSamplesKernel(Out *out, const In *const *in,
                                    int64_t sample_size, int num_samples,
                                    Reduction R = {},
                                    PreprocessorBank *pre = nullptr,
                                    Postprocessor post = {}) {
  OnlineReducer<Acc, Reduction> red;
  int64_t block_size = blockDim.x;
  int64_t grid_size = block_size * gridDim.x;
  for (int64_t ofs = threadIdx.x + blockIdx.x * block_size; ofs < sample_size; ofs += grid_size) {
    red.reset();
    if (pre) {
      for (int i = 0; i < num_samples; i++) {
        auto pp = pre[i].Get({ofs});
        red.add(pp(in[i][ofs]), R);
      }
    } else {
      auto pp = std::remove_reference_t<decltype(pre->Get({ofs}))>();
      for (int i = 0; i < num_samples; i++) {
        red.add(pp(in[i][ofs]), R);
      }
    }
    out[ofs] = ConvertSat<Out>(post(red.result()));
  }
}


/**
 * @brief This function is used for reducing innermost dimension with small extent.
 *
 * The reduction is done in a single pass, with each thread completely reducing
 * the inner dimension.
 *
 * @param pre_bank      preprocessor bank, providing possibly distinct procssing
 *                      per output sample
 * @param post          posptprocessing unary functor
 */
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceInnerSmall(Out *out, const In *in, int64_t n_outer, int n_inner,
                                 Reduction reduce,
                                 PreprocessorBank pre_bank, Postprocessor post) {
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_stride) {
    auto pre = pre_bank.Get({outer});
    const In *base = in + outer * n_inner;
    OnlineReducer<Acc, Reduction> red;
    red.reset();
    for (int i = 0; i < n_inner; i++)
      red.add(pre(__ldg(base + i)), reduce);
    out[outer] = ConvertSat<Out>(post(red.result()));
  }
}

/**
 * @brief This function is used for reducing innermost dimension with medium extent.
 *
 * The reduction is done by a warp.
 * After sequential step, warp reduction is performed.
 */
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceInnerMedium(Out *out, const In *in, int64_t n_outer, int n_inner,
                                  Reduction reduce,
                                  PreprocessorBank pre_bank, Postprocessor post) {
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.y;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  for (int64_t outer = base_idx; outer < n_outer; outer += grid_stride) {
    auto pre = pre_bank.Get({outer});
    const In *base = in + outer * n_inner;
    OnlineReducer<Acc, Reduction> red;
    red.reset();
    for (int i = threadIdx.x; i < n_inner; i += 32)
      red.add(pre(__ldg(base + i)), reduce);
    Acc v = red.result();
    WarpReduce(v, reduce);
    if (threadIdx.x == 0) {
      out[outer] = ConvertSat<Out>(post(v));
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
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceInnerLarge(Out *out, const In *in, int64_t n_outer, int64_t n_inner,
                                 int num_macroblocks, int macroblock_size,
                                 Reduction reduce,
                                 PreprocessorBank pre_bank, Postprocessor post) {
  const int blk_size = 32*blockDim.y;  // block must be warpSize * warpSize for BlockReduce

  const int total_blocks = n_outer * num_macroblocks;

  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;

  int outer_shift = __ffs(num_macroblocks) - 1;
  int inner_mask = num_macroblocks - 1;

  for (int64_t idx = blockIdx.x; idx < total_blocks; idx += gridDim.x) {
    int64_t outer = idx >> outer_shift;
    int64_t inner_macroblock = idx & inner_mask;
    int64_t inner_start = inner_macroblock * macroblock_size;
    int64_t inner_end = cuda_min(n_inner, inner_start + macroblock_size);

    const In *base = &in[outer * n_inner];

    Acc val = reduce.template neutral<Acc>();
    auto pre = pre_bank.Get({outer});

    // reduce macroblock to a block - each thread reduces its own block-strided slice
    bool first = true;
    for (int64_t inner = inner_start + flat_tid; inner < inner_end; inner += blk_size) {
      auto x = pre(__ldg(base + inner));
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
      out[idx] = ConvertSat<Out>(post(val));
  }
}

template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceInner(const ReduceSampleDesc<Out, In> &sample,
                            Reduction reduce,
                            PreprocessorBank pre_bank, Postprocessor post) {
  int64_t n_outer = sample.n_outer;
  int64_t n_reduced = sample.n_reduced;
  Out *out = sample.out;
  const In *in = sample.in;

  if (n_reduced == 1) {
    ReduceFlatNone(out, in, n_outer, pre_bank, post);
  } else if (n_reduced < 32 && sample.num_macroblocks == 1) {
    ReduceInnerSmall<Acc>(out, in, n_outer, n_reduced, reduce, pre_bank, post);
  } else if (n_reduced < 1024 && sample.num_macroblocks == 1) {
    ReduceInnerMedium<Acc>(out, in, n_outer, n_reduced, reduce, pre_bank, post);
  } else {
    ReduceInnerLarge<Acc>(out, in, n_outer, n_reduced,
                          sample.num_macroblocks, sample.macroblock_size,
                          reduce, pre_bank, post);
  }
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceInnerSmallKernel(const ReduceSampleDesc<Out, In> *samples,
                                       Reduction reduce = {},
                                       const PreprocessorBank *pre = nullptr,
                                       const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];
  ReduceInnerSmall<Acc>(sample.out, sample.in, sample.n_outer, sample.n_reduced,
                        reduce, pre_bank, postprocessor);
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceInnerMediumKernel(const ReduceSampleDesc<Out, In> *samples,
                                        Reduction reduce = {},
                                        const PreprocessorBank *pre = nullptr,
                                        const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];
  ReduceInnerMedium<Acc>(sample.out, sample.in, sample.n_outer, sample.n_reduced,
                         reduce, pre_bank, postprocessor);
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceInnerLargeKernel(const ReduceSampleDesc<Out, In> *samples,
                                       Reduction reduce = {},
                                       const PreprocessorBank *pre = nullptr,
                                       const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  auto sample = samples[blockIdx.y];
  ReduceInnerLarge<Acc>(sample.out, sample.in, sample.n_outer, sample.n_reduced,
                        sample.num_macroblocks, sample.macroblock_size,
                        reduce, pre_bank, postprocessor);
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<1>,
          typename Postprocessor = identity>
__global__ void ReduceInnerKernel(const ReduceSampleDesc<Out, In> *samples,
                                  Reduction reduce = {}, const PreprocessorBank *pre = nullptr,
                                  const Postprocessor *post = nullptr) {
  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();
  ReduceInner<Acc>(samples[blockIdx.y], reduce, pre_bank, postprocessor);
}

// Reduction over other dimensions (non-innermost)

/**
 * Each *thread* performs independent, full reduction
 */
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__
void ReduceMiddleSmall(const ReduceSampleDesc<Out, In> &sample,
                       Reduction reduce,
                       PreprocessorBank pre_bank, Postprocessor post) {
  int64_t n_outer = sample.n_outer;
  int n_reduced = sample.n_reduced;
  int64_t n_inner = sample.n_inner;
  int64_t n_non_reduced = n_inner * n_outer;
  int64_t outer_stride = n_inner * n_reduced;

  Out *out = sample.out;
  const In *in = sample.in;
  const int64_t blk_size = blockDim.x * blockDim.y;  // no restriction on block size
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blk_size;
  const int flat_tid = threadIdx.x + threadIdx.y * blockDim.x;
  int64_t base_idx = static_cast<int64_t>(blockIdx.x) * blk_size + flat_tid;
  for (int64_t idx = base_idx; idx < n_non_reduced; idx += grid_stride) {
    int64_t outer, inner;
    outer = idx / n_inner;
    inner = idx % n_inner;
    const In *base = in + outer * outer_stride + inner;
    auto pre = pre_bank.Get({outer, inner});

    OnlineReducer<Acc, Reduction> red;
    red.reset();
    for (int i = 0; i < n_reduced; i++)
      red.add(pre(__ldg(base + i * n_inner)), reduce);

    out[idx] = ConvertSat<Out>(post(red.result()));
  }
}

namespace reduce_shared {
  extern __shared__ uint8_t shared_tmp[];
}  // namespace reduce_shared

/**
 * @brief Reduces the input by fetching contiguous warps of data per block.
 *
 * Each block processes a whole macroblock. Each threads accumulates values
 * assigned to a single output bin in the inner dimension.
 *
 * This function CANNOT work for inner extensts >32.
 * The macroblock size in reduced dimension is limited - see OnlineReducer for limits.
 */
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceMiddleLargeInnerSmall(const ReduceSampleDesc<Out, In> &sample,
                                            Reduction r,
                                            PreprocessorBank pre_bank, Postprocessor post) {
  Acc (*shared_tmp)[33] = reinterpret_cast<Acc (*)[33]>(reduce_shared::shared_tmp);

  Out *out = sample.out;
  const In *in = sample.in;

  const int n_inner = sample.n_inner;
  const int num_macroblocks = sample.num_macroblocks;
  const int macroblock_size = sample.macroblock_size;
  const int64_t macroblock_stride = static_cast<int64_t>(macroblock_size) * n_inner;

  const int64_t total_macroblocks = sample.n_outer * num_macroblocks;

  int outer_shift = __ffs(num_macroblocks) - 1;
  int mblock_mask = num_macroblocks - 1;

  const int warps = blockDim.y;
  const int tid = threadIdx.x + 32 * threadIdx.y;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int blk_size = blockDim.x * blockDim.y;

  const int64_t outer_stride = n_inner * sample.n_reduced;

  // The algorithm:
  // The macroblocks are traversed sequentially with grid stride.
  // The reduced and inner dimensions are flattened and then each warp loads contiguous 32
  // values.
  //
  // Each thread has an accumulator corresponding to one entry in inner dimension - but there
  // may be multiple threads with bins corresponding to the same entry, e.g.
  // for n_inner = 5 the bin assignment is:
  //  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1
  // The assignmnent is not balanced when n_inner is not a divisor of 32, so there are
  // 6 bins for 0, 1 and only for 2, 3, and 4.
  ///
  // For warp 0 in the block, the bins and offsets in the input are aligned. For subsequent
  // warps (and iterations within warp), they are not aligned.
  // warp 1 would see the following indices
  // 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
  // modulo 5 these are
  //  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
  // There are two problems:
  // - the bins the values should go to are no longer aligned with threads' bins.
  // - the bins and values cannot be assigned 1:1, because of the imbalance of bin distribution,
  //   as indicated above.
  // The alignment is realized by shifting the input indices so that the fetched values go
  // into the appropriate bin. The line idx0 = idx + lane_offset does that.
  // Since after applying the offset some values are out of range of the warp's coverage, we need
  // a condition to guard against it (lane + lane_offset < 32).
  // This condition rejects some values, whereas some of the values in warp's coverage were not
  // consumed at all - this is handled by the second fetch from idx1.
  // This ensures that all the values covered by the warp were consumed.
  // The additional fetch does not hurt the performance significantly, since the values are already
  // in L1 cache.
  //
  // The inner loop has block stride, where each warp accumulates values into its own set of bins.
  // After the loop, the per-thread values are stored in shared memory and transposed.
  // Now we can conveniently combine bins gathered by distinct warps and each warp will take care
  // of a subset of bins and store the final reduced result.

  // To avoid the expensive division/modulo in the inner loop, we precalculate some values here:
  const int block_mod = blk_size % n_inner;
  int lane_offset0 = -((warp * 32) % n_inner);
  if (lane_offset0 < 0)
    lane_offset0 += n_inner;

  const int inner_bin = lane % n_inner;

  for (int w = warp; w < 32; w += warps)
    shared_tmp[w][lane] = r.template neutral<Acc>();

  // thread's accumulator
  OnlineReducer<Acc, Reduction> red;

  for (int64_t macroblock = blockIdx.x; macroblock < total_macroblocks; macroblock += gridDim.x) {
    if (macroblock != blockIdx.x)
      __syncthreads();

    // calculate the outer/macroblock coordinates
    int64_t outer = macroblock >> outer_shift;
    int64_t outer_ofs = outer * outer_stride;
    int mblock = macroblock & mblock_mask;
    int64_t macroblock_start = mblock * macroblock_stride;
    int64_t macroblock_end = cuda_min(macroblock_start + macroblock_stride, outer_stride);
    red.reset();
    int lane_offset = lane_offset0;

    auto pre = pre_bank.Get(i64vec2(outer, inner_bin));

    for (int64_t i = macroblock_start; i < macroblock_end; i += blk_size) {
      int64_t idx = i + tid;
      int64_t idx0 = idx + lane_offset;

      Acc v = lane + lane_offset < 32 &&
              idx0 < macroblock_end
                ? pre(__ldg(in + outer_ofs + idx0))
                : r.template neutral<Acc>();
      if (lane + lane_offset >= n_inner && lane < n_inner) {
        int64_t idx1 = idx0 - n_inner;
        if (idx1 < macroblock_end) {
          r(v, pre(__ldg(in + outer_ofs + idx1)));
        }
      }
      red.add(v, r);

      lane_offset -= block_mod;
      if (lane_offset < 0)
        lane_offset += n_inner;
    }

    Acc acc = red.result();  // get the final result from the accumulator...
    shared_tmp[warp][lane] = acc;  // ...and store it in shared memory to be transposed
    __syncthreads();
    // Now the roles of warps and lanes are swapped - lanes now correspond to warps' partial
    // results and warps correspond to bins.
    for (int inner = warp; inner < n_inner; inner += warps) {
      // sum the bins corresponding to given position in inner dimension
      acc = shared_tmp[lane][inner];
      for (int j = inner + n_inner; j < 32; j += n_inner) {
        r(acc, shared_tmp[lane][j]);
      }
      // combine warps' partial results
      WarpReduce(acc, r);
      if (lane == 0) {
        out[macroblock * n_inner + inner] = ConvertSat<Out>(post(acc));
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
template <typename Acc, typename Out, typename In,
          typename Reduction, typename PreprocessorBank, typename Postprocessor>
__device__ void ReduceMiddleLargeInnerMedium(const ReduceSampleDesc<Out, In> &sample,
                                             Reduction r,
                                             PreprocessorBank pre_bank, Postprocessor post) {
  Acc (*shared_tmp)[33] = reinterpret_cast<Acc (*)[33]>(reduce_shared::shared_tmp);

  Out *out = sample.out;
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
    if (out_block != static_cast<int>(blockIdx.x))
      __syncthreads();

    // Caution: fast approximate division ahead!
    int macroblock = __float2int_rd(out_block * rcp_horz_warps);  // integer division - round down
    int horz_warp = out_block - macroblock * horz_warps;  // modulo

    int64_t outer = macroblock >> outer_shift;
    int64_t outer_ofs = outer * outer_stride;
    int mblock = macroblock & mblock_mask;

    int64_t macroblock_start = mblock * macroblock_size;
    int64_t macroblock_end = cuda_min(macroblock_start + macroblock_size, n_reduced);
    OnlineReducer<Acc, Reduction> red;
    red.reset();
    int inner_base = horz_warp * 32;
    int inner = inner_base + lane;
    int64_t base_idx = outer_ofs + inner;
    Acc v;
    // Each warp does a full, parallel reduction over a range of inner dimension.
    if (inner < n_inner) {
      auto pre = pre_bank.Get({ outer, inner });
      for (int64_t i = macroblock_start + warp; i < macroblock_end; i += warps) {
        int64_t offset = base_idx + i * n_inner;
        red.add(pre(__ldg(in + offset)), r);
      }
      v = red.result();
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
        out[macroblock * n_inner + inner] = ConvertSat<Out>(post(v));
      }
    }
  }
}


template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<2>,
          typename Postprocessor = identity>
__global__ void ReduceMiddleKernel(const ReduceSampleDesc<Out, In> *samples,
                                   Reduction reduce = {},
                                   const PreprocessorBank *pre = nullptr,
                                   const Postprocessor *post = nullptr) {
  auto sample = samples[blockIdx.y];

  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();

  if (sample.n_reduced < 1024 && sample.num_macroblocks == 1) {
    ReduceMiddleSmall<Acc>(sample, reduce, pre_bank, postprocessor);
  } else if (sample.n_inner < 32) {
    ReduceMiddleLargeInnerSmall<Acc>(sample, reduce, pre_bank, postprocessor);
  } else {
    ReduceMiddleLargeInnerMedium<Acc>(sample, reduce, pre_bank, postprocessor);
  }
}



template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<2>,
          typename Postprocessor = identity>
__global__ void ReduceMiddleSmallKernel(const ReduceSampleDesc<Out, In> *samples,
                                        Reduction reduce = {},
                                        const PreprocessorBank *pre = nullptr,
                                        const Postprocessor *post = nullptr) {
  auto sample = samples[blockIdx.y];

  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();

  ReduceMiddleSmall<Acc>(sample, reduce, pre_bank, postprocessor);
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<2>,
          typename Postprocessor = identity>
__global__ void ReduceMiddleLargeInnerSmallKernel(const ReduceSampleDesc<Out, In> *samples,
                                                  Reduction reduce = {},
                                                  const PreprocessorBank *pre = nullptr,
                                                  const Postprocessor *post = nullptr) {
  auto sample = samples[blockIdx.y];

  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();

  ReduceMiddleLargeInnerSmall<Acc>(sample, reduce, pre_bank, postprocessor);
}

template <typename Acc, typename Out, typename In,
          typename Reduction = reductions::sum,
          typename PreprocessorBank = reduce_impl::IdentityPreprocessor<2>,
          typename Postprocessor = identity>
__global__ void ReduceMiddleLargeInnerMediumKernel(const ReduceSampleDesc<Out, In> *samples,
                                                   Reduction reduce = {},
                                                   const PreprocessorBank *pre = nullptr,
                                                   const Postprocessor *post = nullptr) {
  auto sample = samples[blockIdx.y];

  PreprocessorBank pre_bank = pre ? pre[blockIdx.y] : PreprocessorBank();
  Postprocessor postprocessor = post ? post[blockIdx.y] : Postprocessor();

  ReduceMiddleLargeInnerMedium<Acc>(sample, reduce, pre_bank, postprocessor);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH_
