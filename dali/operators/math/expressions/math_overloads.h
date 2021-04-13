// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_MATH_OVERLOADS_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_MATH_OVERLOADS_H_

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "dali/core/cuda_utils.h"
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"


#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <cmath>
#endif

namespace dali {

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_sqrt(T x) {
#ifdef __CUDA_ARCH__
  return sqrt(x);
#else
  return std::sqrt(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_rsqrt(T x) {
  return rsqrt(x);
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_cbrt(T x) {
#ifdef __CUDA_ARCH__
  return cbrt(x);
#else
  return std::cbrt(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_exp(T x) {
#ifdef __CUDA_ARCH__
  return exp(x);
#else
  return std::exp(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_log(T x) {
#ifdef __CUDA_ARCH__
  return log(x);
#else
  return std::log(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_log2(T x) {
#ifdef __CUDA_ARCH__
  return log2(x);
#else
  return std::log2(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_log10(T x) {
#ifdef __CUDA_ARCH__
  return log10(x);
#else
  return std::log10(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline std::enable_if_t<std::is_signed<T>::value, T> math_abs(T x) {
  return x < 0 ? -x : x;
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline std::enable_if_t<!std::is_signed<T>::value, T> math_abs(T x) {
  return x;
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_fabs(T x) {
#ifdef __CUDA_ARCH__
  return fabs(x);
#else
  return std::fabs(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_floor(T x) {
#ifdef __CUDA_ARCH__
  return floor(x);
#else
  return std::floor(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_ceil(T x) {
#ifdef __CUDA_ARCH__
  return ceil(x);
#else
  return std::ceil(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_sin(T x) {
#ifdef __CUDA_ARCH__
  return sin(x);
#else
  return std::sin(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_cos(T x) {
#ifdef __CUDA_ARCH__
  return cos(x);
#else
  return std::cos(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_tan(T x) {
#ifdef __CUDA_ARCH__
  return tan(x);
#else
  return std::tan(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_asin(T x) {
#ifdef __CUDA_ARCH__
  return asin(x);
#else
  return std::asin(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_acos(T x) {
#ifdef __CUDA_ARCH__
  return acos(x);
#else
  return std::acos(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_atan(T x) {
#ifdef __CUDA_ARCH__
  return atan(x);
#else
  return std::atan(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_sinh(T x) {
#ifdef __CUDA_ARCH__
  return sinh(x);
#else
  return std::sinh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_cosh(T x) {
#ifdef __CUDA_ARCH__
  return cosh(x);
#else
  return std::cosh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_tanh(T x) {
#ifdef __CUDA_ARCH__
  return tanh(x);
#else
  return std::tanh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_asinh(T x) {
#ifdef __CUDA_ARCH__
  return asinh(x);
#else
  return std::asinh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_acosh(T x) {
#ifdef __CUDA_ARCH__
  return acosh(x);
#else
  return std::acosh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV inline T math_atanh(T x) {
#ifdef __CUDA_ARCH__
  return atanh(x);
#else
  return std::atanh(x);
#endif
}

DALI_NO_EXEC_CHECK
template <typename X, typename Y>
DALI_HOST_DEV inline auto math_pow(
    X x, Y y,
    std::enable_if_t<!std::is_integral<X>::value || !std::is_integral<Y>::value>* = nullptr) {
#ifdef __CUDA_ARCH__
  return pow(x, y);
#else
  return std::pow(x, y);
#endif
}

DALI_NO_EXEC_CHECK
template <typename X, typename Y>
DALI_HOST_DEV std::enable_if_t<std::is_integral<X>::value && std::is_integral<Y>::value,
                               decltype(std::declval<X>() * std::declval<X>())>
math_pow(X x, Y y) {
  return ipow(x, y);
}

// Template special case
DALI_NO_EXEC_CHECK
template <typename X>
DALI_HOST_DEV std::enable_if_t<std::is_integral<X>::value, X>
math_pow(X x, bool y) {
  if (y) {
    return x;
  }
  return 1;
}

DALI_NO_EXEC_CHECK
template <typename X, typename Y>
DALI_HOST_DEV inline auto math_atan2(X x, Y y) {
#ifdef __CUDA_ARCH__
  return atan2(x, y);
#else
  return std::atan2(x, y);
#endif
}


}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_MATH_OVERLOADS_H_
