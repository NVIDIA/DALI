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

#ifndef _DALI_KERNELS_REDUCE_REDUCE_INLINE_CUH
#define _DALI_KERNELS_REDUCE_REDUCE_INLINE_CUH

#include <cuda_runtime.h>
#include "dali/core/host_dev.h"

namespace dali {
namespace kernels {
namespace reductions {


template <int n, typename Acc>
struct StaticTreeReduce {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, int64_t stride, Reduction r, Preprocess pp) {
    const int b = n / 2;
    const int a = n - b;
    Acc acc = StaticTreeReduce<a, Acc>::reduce(base, stride, r, pp);
    r(acc, StaticTreeReduce<b, Acc>::reduce(base + a * stride, stride, r, pp));
    return acc;
  }
};

template <typename Acc>
struct StaticTreeReduce<0, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, int64_t stride, Reduction r, Preprocess pp) {
    return r.template neutral<Acc>();
  }
};

template <typename Acc>
struct StaticTreeReduce<1, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, int64_t stride, Reduction, Preprocess pp) {
  #ifdef __CUDA_ARCH__
    return pp(__ldg(base));
  #else
    return pp(*base);
  #endif
  }
};

template <typename Acc>
struct StaticTreeReduce<2, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, int64_t stride, Reduction r, Preprocess pp) {
  #ifdef __CUDA_ARCH__
    Acc acc = pp(__ldg(base));
    r(acc, pp(__ldg(base + stride)));
  #else
    Acc acc = pp(base[0]);
    r(acc, pp(base[stride]));
  #endif
    return acc;
  }
};

template <typename Acc, typename T, typename Reduction, typename Preprocess>
DALI_HOST_DEV Acc ThreadReduce(const T *base, int n, int64_t stride, Reduction r, Preprocess pp) {
  Acc acc = r.template neutral<Acc>();
  while (n >= 256) {
    r(acc, StaticTreeReduce<256, Acc>::reduce(base, stride, r, pp));
    base += 256 * stride;
    n -= 256;
  }
  #define INLINE_TREE_REDUCE_STEP(pow) \
  if (n >= (1 << pow)) { \
    r(acc, StaticTreeReduce<(1 << pow), Acc>::reduce(base, stride, r, pp)); \
    base += (1 << pow) * stride; \
    n -= (1 << pow); \
  }
  INLINE_TREE_REDUCE_STEP(7)
  INLINE_TREE_REDUCE_STEP(6)
  INLINE_TREE_REDUCE_STEP(5)
  INLINE_TREE_REDUCE_STEP(4)
  INLINE_TREE_REDUCE_STEP(3)
  // below 8, we handle all cases, in a switch

  switch (n) {
    case 1:
    #ifdef __CUDA_ARCH__
      r(acc, pp(__ldg(base)));
    #else
      r(acc, pp(*base));
    #endif
      break;
    case 2:
      r(acc, StaticTreeReduce<2, Acc>::reduce(base, stride, r, pp));
      break;
    case 3:
      r(acc, StaticTreeReduce<3, Acc>::reduce(base, stride, r, pp));
      break;
    case 4:
      r(acc, StaticTreeReduce<4, Acc>::reduce(base, stride, r, pp));
      break;
    case 5:
      r(acc, StaticTreeReduce<5, Acc>::reduce(base, stride, r, pp));
      break;
    case 6:
      r(acc, StaticTreeReduce<6, Acc>::reduce(base, stride, r, pp));
      break;
    case 7:
      r(acc, StaticTreeReduce<7, Acc>::reduce(base, stride, r, pp));
      break;
    default:
      break;  // no-op
  }
  return acc;
}

}  // namespace reductions
}  // namespace kernels
}  // namespace dali


#endif  // _DALI_KERNELS_REDUCE_REDUCE_INLINE_CUH
