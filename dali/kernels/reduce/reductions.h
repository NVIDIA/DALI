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

#ifndef DALI_KERNELS_REDUCE_REDUCTIONS_H_
#define DALI_KERNELS_REDUCE_REDUCTIONS_H_

#include <cassert>
#include <utility>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/util.h"
#include "dali/core/convert.h"
#include "dali/core/geom/vec.h"
#include "dali/core/cuda_utils.h"

namespace dali {
namespace kernels {
namespace reductions {

struct square {
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(const T &x) const noexcept {
    return x * x;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  int64_t operator()(int32_t x) const noexcept {
    return static_cast<int64_t>(x) * x;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  uint64_t operator()(uint32_t x) const noexcept {
    return static_cast<uint64_t>(x) * x;
  }
};

template <typename Mean>
struct variance {
  Mean mean = 0;
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(const T &x) const noexcept {
    auto d = x - mean;
    return d * d;
  }
};

struct sum {
  template <typename Acc, typename Addend>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(Acc &acc, const Addend &val) const noexcept {
    acc += val;
  }

  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr T neutral() noexcept { return 0; }
};

template <typename T>
struct min_impl {
  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static T reduce(T &min_val, const U &val) noexcept {
    return val < min_val ? val : min_val;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr T neutral() noexcept { return max_value<T>(); }
};

template <typename T, int N>
struct min_impl<vec<N, T>> {
  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static vec<N, T> reduce(vec<N, T> &min_val, const vec<N, U> &val) noexcept {
    IMPL_VEC_ELEMENTWISE(min_impl<T>::reduce(min_val[i], val[i]));
  }

  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static vec<N, T> reduce(vec<N, T> &min_val, const U &val) noexcept {
    IMPL_VEC_ELEMENTWISE(min_impl<T>::reduce(min_val[i], val));
  }


  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr vec<N, T> neutral() noexcept { return max_value<T>(); }
};

template <typename T>
struct max_impl {
  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static T reduce(T &max_val, const U &val) noexcept {
    return val > max_val ? val : max_val;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr T neutral() noexcept { return min_value<T>(); }
};

template <typename T, int N>
struct max_impl<vec<N, T>> {
  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static vec<N, T> reduce(vec<N, T> &max_val, const vec<N, U> &val) noexcept {
    IMPL_VEC_ELEMENTWISE(max_impl<T>::reduce(max_val[i], val[i]));
  }

  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  static vec<N, T> reduce(vec<N, T> &max_val, const U &val) noexcept {
    IMPL_VEC_ELEMENTWISE(max_impl<T>::reduce(max_val[i], val));
  }


  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr vec<N, T> neutral() noexcept { return min_value<T>(); }
};

struct min {
  template <typename T, typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(T &min_val, const U &val) const noexcept {
    min_val = min_impl<T>::reduce(min_val, val);
  }

  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr T neutral() noexcept { return min_impl<T>::neutral(); }
};

struct max {
  template <typename T, typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(T &max_val, const U &val) const noexcept {
    max_val = max_impl<T>::reduce(max_val, val);
  }

  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  static constexpr T neutral() noexcept { return max_impl<T>::neutral(); }
};

template <typename Reduction>
struct is_accurate : std::false_type {};

template <>
struct is_accurate<min> : std::true_type {};
template <>
struct is_accurate<max> : std::true_type {};

template <typename Reduction>
constexpr bool IsAccurate(const Reduction &) { return is_accurate<Reduction>::value; }

}  // namespace reductions
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCTIONS_H_
