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
Out RefReduce(span<T> in, const Reduction &R) {
  switch (in.size()) {
    case 0:
      return R.template neutral<Out>();
    case 1:
      return in[0];
    default: {
      if (in.size() <= 128) {
        double acc = R.template neutral<Out>();
        for (auto &x : in)
          R(acc, x);
        return acc;
      }
      int64_t m = in.size() / 2;
      int64_t n = in.size() - m;
      Out out = RefReduce<Out>(make_span(in.data(), m), R);
      R(out, RefReduce<Out>(make_span(in.data() + m, n), R));
      return out;
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_TEST_H_
