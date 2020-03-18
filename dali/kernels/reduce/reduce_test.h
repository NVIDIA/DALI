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

#ifndef DALI_KERNELS_REDUCE_REDUCE_TEST_H_
#define DALI_KERNELS_REDUCE_REDUCE_TEST_H_

#include "dali/core/span.h"

namespace dali {
namespace kernels {

constexpr bool IsAccurate(const reductions::min &) { return true; }
constexpr bool IsAccurate(const reductions::max &) { return true; }
template <typename Reduction>
constexpr bool IsAccurate(const Reduction &) { return false; }


template <typename Out, typename Reduction, typename T>
Out RefReduce(const T *in, int64_t n, int64_t stride, const Reduction &R) {
  switch (n) {
    case 0:
      return R.template neutral<Out>();
    case 1:
      return in[0];
    default: {
      if (n <= 128) {
        Out acc = R.template neutral<Out>();
        for (int64_t idx = 0; idx < n; idx++)
          R(acc, in[idx * stride]);
        return acc;
      }
      int64_t a = n / 2;
      int64_t b = n - a;
      Out out = RefReduce<Out>(in, a, stride, R);
      R(out, RefReduce<Out>(in + a * stride, b, stride, R));
      return out;
    }
  }
}


template <typename Out, typename Reduction, typename T>
Out RefReduce(span<T> in, const Reduction &R) {
  return RefReduce<Out>(in.data(), in.size(), 1, R);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_TEST_H_
