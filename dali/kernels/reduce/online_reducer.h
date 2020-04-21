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

#ifndef DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_
#define DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_

#include "dali/kernels/reduce/reductions.h"

namespace dali {
namespace kernels {

template <typename Acc, typename Reduction>
struct TrivialReducer {
  Acc value;

  DALI_HOST_DEV DALI_FORCEINLINE void reset() {
    value = Reduction::template neutral<Acc>();
  }

  DALI_HOST_DEV DALI_FORCEINLINE void add(Acc x, Reduction r = {}) {
    r(value, x);
  }

  DALI_HOST_DEV DALI_FORCEINLINE Acc result() const { return value; }
};

template <typename Acc, typename Reduction>
struct OnlineReducer : TrivialReducer<Acc, Reduction> {};

/**
 * @brief Implements compensated sum.
 *
 * The residue contains accumulated error, effectively doubling the precision.
 */
template <typename Acc, bool is_fp = std::is_floating_point<Acc>::value>
struct OnlineSum {
  Acc sum, residue;

  DALI_HOST_DEV DALI_FORCEINLINE void reset() {
    sum = residue = 0;
  }

  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  void add(T value, reductions::sum = {}) {
  #ifdef __CUDA_ARCH__
    // protect against fast_math optimizations
    Acc addend = __fadd_rn(residue, value);
    Acc new_sum = __fadd_rn(sum, addend);
    residue = __fadd_rn(residue, __fsub_rn(value, __fsub_rn(new_sum, sum)));
    sum = new_sum;
  #else
    Acc new_sum = sum + (residue + value);
    residue += value - (new_sum - sum);
    sum = new_sum;
  #endif
  }

  DALI_HOST_DEV DALI_FORCEINLINE Acc result() const { return sum; }
};

template <typename Acc>
struct OnlineSum<Acc, false> : TrivialReducer<Acc, reductions::sum> {};

template <typename Acc>
struct OnlineReducer<Acc, reductions::sum> : OnlineSum<Acc> {};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_
