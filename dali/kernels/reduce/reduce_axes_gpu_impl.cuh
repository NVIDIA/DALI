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
 * @brief This kernel is used for reducing innermost dimension with small (<32) extent.
 *
 * Limitations: innermost extent up to 64 elements
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceInnerSmallKernel(Acc *out, const In *const *in, const int64_t *in_shapes,
                                       Reduction reduce = {}, Preprocess pp = {}) {

  auto bank_hack = [](int index) {
    // value of 19 gives no bank conflicts for inner dimensions 1 to 8 and is only really bad for
    // inner dim = 27 and 47, when it uses only 4 out of 16 banks
    return (index & 1) + (index >> 4) * 19;
  };

  __shared__ Acc tmp[64*38];  // 38 = 32 * 19/16, see bank_hack

  int sample = threadIdx.y;
  int64_t n_outer = in_shapes[2*samples];
  int n_inner = in_shapes[2*sample+1];
  int64_t sample_size = n_outer * n_inner;

  int64_t grid_size_x = blockDim.x * gridDim.x;
  int64_t grid_stride = grid_size_x * n_inner;
  int64_t block_stride = blockDim.x * n_inner;
  int64_t block_start = blockIdx.x * block_stride;

  for (int64_t base = block_start; base < sample_size; base += block_stride) {
    for (int i = 0; i < n_inner; i++) {
      int ofs = i * blockDim.x + threadIdx.x;
      auto idx = base + ofs;
      tmp[bank_hack(ofs)] = idx < sample_size ? pp(in[sample][idx]) : R.template neutral<Acc>();
    }
    __syncthreads();
    Acc acc = tmp[bank_hack(threadIdx.x * n_inner)];
    for (int i = 1; i < n_inner; i++) {
      int ofs = threadIdx.x * n_inner + i;
      reduce(acc, tmp[bank_hack(ofs)]);
    }
    __syncthreads()
    out[blockIdx.x + gridDim.x * blockIdx.y] = acc;
  }
}

/**
 * @brief This kernel is used for reducing innermost dimension with medium extent.
 *
 * Limitations: innermost extent 32 to ~256 elements
 */
template <typename Acc, typename In,
          typename Reduction = reductions::sum,
          typename Preprocess = dali::identity>
__global__ void ReduceInnerMediumKernel(Acc *out, const In *const *in, const int64_t *in_sizes,
                                       Reduction reduce = {}, Preprocess pp = {}) {
  Acc val = reduce.template neutral<Acc>();

  int sample = threadIdx.y;
  int64_t n_outer = in_shapes[2*samples];
  int n_inner = in_shapes[2*sample+1];
  int64_t sample_size = n_outer * n_inner;

  int64_t grid_size_x = blockDim.x * gridDim.x;
  int64_t grid_stride = grid_size_x * n_inner;
  int64_t block_stride = blockDim.x * n_inner;
  int64_t block_start = blockIdx.x * block_stride;

}

}  // namespace kernels
}  // namespace dali

#endif  // _DALI_KERNELS_REDUCE_REDUCE_AXES_GPU_IMPL_CUH
