// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_FLOAT16_H_
#define DALI_CORE_FLOAT16_H_

#include <cuda_fp16.h>  // for __half & related methods
#include <type_traits>
#ifndef __CUDA_ARCH__
#include "dali/util/half.hpp"
#endif

namespace dali {

// For the GPU
#ifdef __CUDA_ARCH__
using float16 = __half;
#else
using float16 = half_float::half;
#endif

namespace detail {

template <typename T>
struct is_half : std::false_type {};

template <>
struct is_half<float16> : std::true_type {};

}  // namespace detail

template <typename T>
struct is_half : detail::is_half<std::remove_cv_t<T>> {};

template <typename T>
struct is_arithmetic_or_half {
  static constexpr bool value =
    std::is_arithmetic<T>::value || is_half<T>::value;
};

template <typename T>
struct is_fp_or_half {
  static constexpr bool value =
    std::is_floating_point<T>::value || is_half<T>::value;
};


}  // namespace dali

#endif  // DALI_CORE_FLOAT16_H_
