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

#ifndef DALI_CORE_CONVERT_H_
#define DALI_CORE_CONVERT_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>
#ifndef __CUDA_ARCH__
#include "dali/util/half.hpp"
#else
#include "dali/core/cuda_utils.h"
#endif
namespace dali {

template <typename T>
struct const_limits;

// std::numeric_limits are not compatible with CUDA
template <typename T>
__host__ __device__ constexpr T max_value() {
  return const_limits<std::remove_cv_t<T>>::max;
}
template <typename T>
__host__ __device__ constexpr T min_value() {
  return const_limits<std::remove_cv_t<T>>::min;
}

#define DEFINE_TYPE_RANGE(type, min_, max_) template <>\
struct const_limits<type> { static constexpr type min = min_, max = max_; }

DEFINE_TYPE_RANGE(bool, false, true);
DEFINE_TYPE_RANGE(uint8_t,  0, 0xff);
DEFINE_TYPE_RANGE(int8_t,  -0x80, 0x7f);
DEFINE_TYPE_RANGE(uint16_t, 0, 0xffff);
DEFINE_TYPE_RANGE(int16_t, -0x8000, 0x7fff);
DEFINE_TYPE_RANGE(uint32_t, 0, 0xffffffff);
DEFINE_TYPE_RANGE(int32_t, -0x80000000, 0x7fffffff);
DEFINE_TYPE_RANGE(uint64_t, 0, 0xffffffffffffffffUL);
DEFINE_TYPE_RANGE(int64_t, -0x8000000000000000L, 0x7fffffffffffffffL);
DEFINE_TYPE_RANGE(float, -3.40282347e+38f, 3.40282347e+38f);
DEFINE_TYPE_RANGE(double, -1.7976931348623157e+308, 1.7976931348623157e+308);

template <typename From, typename To>
struct needs_clamp {
  static constexpr bool from_fp = std::is_floating_point<From>::value;
  static constexpr bool to_fp = std::is_floating_point<To>::value;
  static constexpr bool from_unsigned = std::is_unsigned<From>::value;
  static constexpr bool to_unsigned = std::is_unsigned<To>::value;

  static constexpr bool value =
    // to smaller type of same kind (fp, int)
    (from_fp == to_fp && sizeof(To) < sizeof(From)) ||
    // fp32 has range in excess of (u)int64
    (from_fp && !to_fp) ||
    // converting to unsigned requires clamping negatives to zero
    (!from_unsigned && to_unsigned) ||
    // zero-extending signed unsigned integers requires more bits
    (from_unsigned && !to_unsigned && sizeof(To) <= sizeof(From));
};

template <typename T>
struct ret_type {  // a placeholder for return type
  constexpr ret_type() = default;
};

template <typename T, typename U>
__host__ __device__ constexpr std::enable_if_t<
    needs_clamp<U, T>::value && std::is_signed<U>::value,
    T>
clamp(U value, ret_type<T>) {
  return value < min_value<T>() ? min_value<T>() :
         value > max_value<T>() ? max_value<T>() :
         static_cast<T>(value);
}

template <typename T, typename U>
__host__ __device__ constexpr std::enable_if_t<
    needs_clamp<U, T>::value && std::is_unsigned<U>::value,
    T>
clamp(U value, ret_type<T>) {
  return value > max_value<T>() ? max_value<T>() : static_cast<T>(value);
}

template <typename T, typename U>
__host__ __device__ constexpr std::enable_if_t<
    !needs_clamp<U, T>::value,
    T>
clamp(U value, ret_type<T>) { return value; }

__host__ __device__ constexpr int32_t clamp(uint32_t value, ret_type<int32_t>) {
  return value & 0x80000000u ? 0x7fffffff : value;
}

__host__ __device__ constexpr uint32_t clamp(int32_t value, ret_type<uint32_t>) {
  return value < 0 ? 0u : value;
}

__host__ __device__ constexpr int32_t clamp(int64_t value, ret_type<int32_t>) {
  return value < static_cast<int64_t>(min_value<int32_t>()) ? min_value<int32_t>() :
         value > static_cast<int64_t>(max_value<int32_t>()) ? max_value<int32_t>() :
         static_cast<int32_t>(value);
}

template <>
__host__ __device__ constexpr int32_t clamp(uint64_t value, ret_type<int32_t>) {
  return value > static_cast<uint64_t>(max_value<int32_t>()) ? max_value<int32_t>() :
         static_cast<int32_t>(value);
}

template <>
__host__ __device__ constexpr uint32_t clamp(int64_t value, ret_type<uint32_t>) {
  return value < 0 ? 0 :
         value > static_cast<int64_t>(max_value<uint32_t>()) ? max_value<uint32_t>() :
         static_cast<uint32_t>(value);
}

template <>
__host__ __device__ constexpr uint32_t clamp(uint64_t value, ret_type<uint32_t>) {
  return value > static_cast<uint64_t>(max_value<uint32_t>()) ? max_value<uint32_t>() :
         static_cast<uint32_t>(value);
}

template <typename T>
__host__ __device__ constexpr bool clamp(T value, ret_type<bool>) {
  return static_cast<bool>(value);
}

#ifndef __CUDA_ARCH__
template <typename T>
__host__ __device__ constexpr half_float::half clamp(T value, ret_type<half_float::half>) {
  return static_cast<half_float::half>(value);
}

template <typename T>
__host__ __device__ constexpr T clamp(half_float::half value, ret_type<T>) {
  return clamp(static_cast<float>(value), ret_type<T>());
}

__host__ __device__ inline bool clamp(half_float::half value, ret_type<bool>) {
  return static_cast<bool>(value);
}

__host__ __device__ constexpr half_float::half clamp(half_float::half value,
                                                     ret_type<half_float::half>) {
  return value;
}

#else

template <typename T>
__host__ __device__ constexpr float16 clamp(T value, ret_type<float16>) {
  return static_cast<float16>(value);
}

// __half does not have a constructor for int64_t, use long long
__host__ __device__ inline float16 clamp(int64_t value, ret_type<float16>) {
  return static_cast<float16>(static_cast<long long int>(value));  // NOLINT
}

template <typename T>
__host__ __device__ constexpr T clamp(float16 value, ret_type<T>) {
  return clamp(static_cast<float>(value), ret_type<T>());
}

__host__ __device__ inline bool clamp(float16 value, ret_type<bool>) {
  return static_cast<bool>(value);
}

__host__ __device__ constexpr float16 clamp(float16 value, ret_type<float16>) {
  return value;
}

#endif

template <typename T, typename U>
__host__ __device__ constexpr T clamp(U value) {
  return clamp(value, ret_type<T>());
}

template <typename Out, typename In>
__host__ __device__ constexpr Out Convert(In value) {
  return static_cast<Out>(value);
}

template <typename Out, typename In>
__host__ __device__ constexpr Out ConvertSat(In value) {
  return clamp<Out>(value);
}

template <typename Out, typename In>
__host__ __device__ constexpr std::enable_if_t<
    std::is_floating_point<Out>::value && !std::is_floating_point<In>::value, Out>
ConvertNorm(In value) {
  return value * (Out(1) / max_value<In>());
}

template <typename Out, typename In>
__host__ __device__ constexpr std::enable_if_t<
    std::is_floating_point<Out>::value && std::is_floating_point<In>::value, Out>
ConvertNorm(In value) {
  return static_cast<Out>(value);
}

template <typename Out>
constexpr __device__ __host__ std::enable_if_t<std::is_unsigned<Out>::value, Out>
ConvertSatNorm(float value) {
#ifdef __CUDA_ARCH__
  return max_value<Out>() * __saturatef(value);
#else
  return max_value<Out>() * (value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value);
#endif
}

template <typename Out>
constexpr __device__ __host__ std::enable_if_t<
  std::is_signed<Out>::value && std::is_integral<Out>::value, Out>
ConvertSatNorm(float value) {
  return clamp<Out>(value * static_cast<float>(max_value<Out>()));
}

template <typename Out, typename In>
__host__ __device__ constexpr std::enable_if_t<
    !std::is_floating_point<Out>::value && std::is_floating_point<In>::value, Out>
ConvertNorm(In value) {
  return ConvertSatNorm<Out>(value);
}

}  // namespace dali

#endif  // DALI_CORE_CONVERT_H_
