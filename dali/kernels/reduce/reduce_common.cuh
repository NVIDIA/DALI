// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_REDUCE_COMMON_CUH_
#define DALI_KERNELS_REDUCE_REDUCE_COMMON_CUH_

#include <cuda_runtime.h>
#include "dali/core/util.h"

namespace dali {
namespace kernels {

template <typename T>
DALI_FORCEINLINE __device__ T shfl_down(T &t, int n) {
  constexpr unsigned FULL_MASK = 0xffffffffu;
  return __shfl_down_sync(FULL_MASK, t, n);
}

template <typename T, int N>
DALI_FORCEINLINE __device__ vec<N, T> shfl_down(vec<N, T> &t, int n) {
  constexpr unsigned FULL_MASK = 0xffffffffu;
  IMPL_VEC_ELEMENTWISE(__shfl_down_sync(FULL_MASK, t[i], n));
}

template <typename Acc, typename Reduction>
DALI_FORCEINLINE __device__ void WarpReduce(Acc &val, Reduction reduce) {
  reduce(val, shfl_down(val, 16));
  reduce(val, shfl_down(val, 8));
  reduce(val, shfl_down(val, 4));
  reduce(val, shfl_down(val, 2));
  reduce(val, shfl_down(val, 1));
}

/**
 * @brief Reduces `val` across threads in a block
 *
 * First, `val` is reduced using warp shuffle.
 * The block width is 32 and the thread 0 in each warp stores its result in shared memory.
 * After block synchronization, only the first warp does anything - it loads the stored
 * values into respective lanes and does one more warp reduction.
 * Finally, the thread (0, 0) stores thre result at given index.
 *
 * @remarks blockDim must be (32, pow2), where pow2 is <= 32
 * @remarks Note: blockDim is expected to be 2D, but that doesn't mean the data must be 2D.
 *          Users of BlockReduce would typically use a flat thread id to access the data.
 * @see     ReduceAllKernel for a usage example
 */
template <typename Acc, typename Reduction>
__device__ bool BlockReduce(Acc &val, Reduction reduce) {
  __shared__ Acc tmp[32];
  WarpReduce(val, reduce);
  if (threadIdx.x == 0)
    tmp[threadIdx.y] = val;
  else if (threadIdx.x >= blockDim.y)  // fill missing elements with neutrals so we can warp-reduce
    tmp[threadIdx.x] = reduce.template neutral<Acc>();
  __syncthreads();

  if (threadIdx.y == 0) {
    val = tmp[threadIdx.x];
    // we don't really reduce entire warp if blockDim.y < 32, but it's properly
    // padded with neutral elements and checking for proper number of lanes
    // would likely be more expensive than one or two extra __shfl_down_sync
    WarpReduce(val, reduce);
    return threadIdx.x == 0;
  }
  return false;
}

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_REDUCE_REDUCE_COMMON_CUH_
