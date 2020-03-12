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

namespace dali {
namespace kernels {

/**
 * @brief This function is used for reducing innermost dimension with small (<32) extent.
 *
 * Limitations: innermost extent up to 63 elements
 *
 * @param out per-sample output (partial result)
 * @param in  input sample (single tensor)
 * @param
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerSmall(Acc *out, const In *in, int64_t n_outer, int n_inner,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  // this table contains multipliers to use in shared memory reindexing to best avoid bank
  // conflicts - the values are result of exhaustive search for particular
  static __constant__ uint8_t bank_friendly_table[64] = {
    17, 16, 17, 16, 17, 16, 19, 16,
    17, 16, 19, 16, 19, 16, 17, 16,
    17, 16, 17, 16, 19, 16, 19, 16,
    17, 16, 17, 16, 19, 16, 19, 16,
    17, 16, 17, 16, 17, 16, 17, 16,
    17, 16, 17, 16, 17, 16, 17, 16,
    17, 16, 17, 16, 17, 16, 17, 16,
    17, 16, 17, 16, 19, 16, 17, 16,
  };

  __shared__ Acc tmp[64*38];  // 38 = 32 * 19/16, see bank_friendly

  int64_t sample_size = n_outer * n_inner;

  int bank_friendly_mul = bank_friendly_table[n_inner];
  auto bank_friendly = [bank_friendly_mul](int index) {
    // value of 19 gives no bank conflicts for inner dimensions 1 to 8 and is only really bad for
    // inner dim = 27 and 47, when it uses only 4 out of 16 banks
    return (index & 1) + (index >> 4) * bank_friendly_mul;
  };

  int64_t grid_size_x = blockDim.x * gridDim.x;
  int64_t grid_stride = grid_size_x * n_inner;

  for (int64_t outer = blockIdx.x * blockDim.x; outer < n_outer; outer += grid_size_x) {
    int64_t base = outer * n_inner;
    for (int i = 0; i < n_inner; i++) {
      int ofs = i * blockDim.x + threadIdx.x;
      auto idx = base + ofs;
      tmp[bank_friendly(ofs)] = idx < sample_size ? pp(in[idx]) : reduce.template neutral<Acc>();
    }
    __syncthreads();
    int64_t outer_ofs = outer + threadIdx.x;
    Acc acc;
    if (outer_ofs < n_outer) {
      Acc acc = tmp[bank_friendly(threadIdx.x * n_inner)];
      for (int i = 1; i < n_inner; i++) {
        int ofs = threadIdx.x * n_inner + i;
        reduce(acc, tmp[bank_friendly(ofs)]);
      }
    }
    __syncthreads();  // this must be in convergent code
    if (outer_ofs < n_outer)
      out[outer_ofs] = acc;
  }
}

/**
 * @brief This kernel is used for reducing innermost dimension with medium extent.
 *
 * Limitations: innermost extent 32 to ~256 elements due to performance and precision issues
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerMedium(Acc *out, const In *in, int64_t n_outer, int n_inner,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  Acc val = reduce.template neutral<Acc>();
}

/**
 * @brief This kernel is used for reducing innermost dimension with large extent.
 *
 * Limitations: not recommended for small reduced extents due to performance.
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__device__ void ReduceInnerLarge(Acc *out, const In *in, int64_t n_outer, int64_t n_inner,
                                  Reduction reduce = {}, Preprocess pp = {}) {
  Acc val = reduce.template neutral<Acc>();
}

template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceInnerKernel(Acc *const *out, const In *const *in, const int64_t *in_shapes,
                                 Reduction reduce = {}, Preprocess pp = {}) {
  int sample = threadIdx.y;
  int64_t n_outer = in_shapes[2*sample];
  int64_t n_inner = in_shapes[2*sample+1];
  int64_t sample_size = n_outer * n_inner;

  if (n_inner < 64) {
    ReduceInnerSmall(&out[sample], in[sample], n_outer, n_inner, reduce, pp);
  } else if (n_inner < 256) {
    ReduceInnerMedium(&out[sample], in[sample], n_outer, n_inner, reduce, pp);
  } else {
    ReduceInnerLarge(&out[sample], in[sample], n_outer, n_inner, reduce, pp);
  }

}

}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
