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
struct TreeReduce {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, Reduction r, Preprocess pp) {
    const int b = n / 2;
    const int a = n - b;
    Acc acc = TreeReduce<a, Acc>::reduce(base, r, pp);
    r(acc, TreeReduce<b, Acc>::reduce(base + a, r, pp));
    return acc;
  }
};

template <typename Acc>
struct TreeReduce<0, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, Reduction r, Preprocess pp) {
    return r.template neutral<Acc>();
  }
};

template <typename Acc>
struct TreeReduce<1, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, Reduction, Preprocess pp) {
    return pp(__ldg(base));
  }
};

template <typename Acc>
struct TreeReduce<2, Acc> {
  template <typename T, typename Reduction, typename Preprocess>
  DALI_FORCEINLINE DALI_HOST_DEV
  static Acc reduce(const T *base, Reduction r, Preprocess pp) {
    Acc acc = pp(__ldg(base));
    r(acc, pp(__ldg(base+1)));
    return acc;
  }
};

template <typename Acc, typename T, typename Reduction, typename Preprocess>
DALI_HOST_DEV Acc ReduceInner(const T *base, int n, Reduction r, Preprocess pp) {
  Acc acc = r.template neutral<Acc>();
  while (n >= 256) {
    r(acc, TreeReduce<256, Acc>::reduce(base, r, pp));
    base += 256;
    n -= 256;
  }
  if (n >= 128) {
    r(acc, TreeReduce<128, Acc>::reduce(base, r, pp));
    base += 128;
    n -= 128;
  }
  if (n >= 64) {
    r(acc, TreeReduce<64, Acc>::reduce(base, r, pp));
    base += 64;
    n -= 64;
  }
  if (n >= 32) {
    r(acc, TreeReduce<32, Acc>::reduce(base, r, pp));
    base += 32;
    n -= 32;
  }
  if (n >= 16) {
    r(acc, TreeReduce<16, Acc>::reduce(base, r, pp));
    base += 16;
    n -= 16;
  }
  if (n >= 8) {
    r(acc, TreeReduce<8, Acc>::reduce(base, r, pp));
    base += 8;
    n -= 8;
  }
  switch (n) {
    case 0:
      break;
    case 1:
      r(acc, pp(*base));
      break;
    case 2:
      r(acc, TreeReduce<2, Acc>::reduce(base, r, pp));
      break;
    case 3:
      r(acc, TreeReduce<3, Acc>::reduce(base, r, pp));
      break;
    case 4:
      r(acc, TreeReduce<4, Acc>::reduce(base, r, pp));
      break;
    case 5:
      r(acc, TreeReduce<5, Acc>::reduce(base, r, pp));
      break;
    case 6:
      r(acc, TreeReduce<6, Acc>::reduce(base, r, pp));
      break;
    case 7:
      r(acc, TreeReduce<7, Acc>::reduce(base, r, pp));
      break;
  }
  return acc;
}

}  // namespace reductions
}  // namespace kernels
}  // namespace dali


#endif  // _DALI_KERNELS_REDUCE_REDUCE_INLINE_CUH
