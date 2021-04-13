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

#ifndef DALI_CORE_MATH_UTIL_H_
#define DALI_CORE_MATH_UTIL_H_

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <cmath>
#endif
#include <cstdint>
#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"

namespace dali {

/// @brief Round down and convert to integer
inline DALI_HOST_DEV int floor_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rd(x);
#else
  return static_cast<int>(std::floor(x));
#endif
}

/// @brief Round up and convert to integer
inline DALI_HOST_DEV int ceil_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_ru(x);
#else
  return static_cast<int>(std::ceil(x));
#endif
}

/// @brief Round to nearest integer and convert
inline DALI_HOST_DEV int round_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rn(x);
#else
  return static_cast<int>(std::roundf(x));
#endif
}

template <typename T>
DALI_HOST_DEV constexpr T clamp(const T &value, const T &lo, const T &hi) {
  // Going through intermediate value saves us a jump
  #ifdef __CUDA_ARCH__
    return min(hi, max(value, lo));
  #else
    T x = value < lo ? lo : value;
    return x > hi ? hi : x;
  #endif
}

/// @brief Calculate square root reciprocal.
DALI_HOST_DEV inline float rsqrt(float x) {
#ifdef __CUDA_ARCH__
  return __frsqrt_rn(x);
#elif defined __SSE__
  // Use SSE intrinsic and one Newton-Raphson refinement step
  // - faster and less hacky than the hack below.
  __m128 X = _mm_set_ss(x);
  __m128 tmp = _mm_rsqrt_ss(X);
  float y = _mm_cvtss_f32(tmp);
  return y * (1.5f - x*0.5f * y*y);
#else
  // Fallback to bit-level hacking.
  // https://en.wikipedia.org/wiki/Fast_inverse_square_root
  int32_t i;
  float x2, y;
  x2 = x * 0.5f;
  y  = x;
  i  = *(const int32_t*)&y;
  i  = 0x5F375A86 - (i >> 1);
  y  = *(const float *)&i;
  // Two Newton-Raphson steps gives 5-6 significant digits
  y  = y * (1.5f - (x2 * y * y));
  y  = y * (1.5f - (x2 * y * y));
  return y;
#endif
}

/// @brief Calculate fast approximation of square root reciprocal.
DALI_HOST_DEV inline float fast_rsqrt(float x) {
#ifdef __CUDA_ARCH__
  return __frsqrt_rn(x);
#elif defined __SSE__
  // Use SSE intrinsic.
  // - without the refinement step, it's much faster and less hacky than the hack below.
  __m128 X = _mm_set_ss(x);
  __m128 tmp = _mm_rsqrt_ss(X);
  return _mm_cvtss_f32(tmp);
#else
  // Fallback to bit-level hacking.
  // https://en.wikipedia.org/wiki/Fast_inverse_square_root
  int32_t i;
  float x2, y;
  x2 = x * 0.5f;
  y  = x;
  i  = *(const int32_t*)&y;
  i  = 0x5F375A86 - (i >> 1);
  y  = *(const float *)&i;
  // One Newton-Raphson step gives 3 significant digits
  y  = y * (1.5f - (x2 * y * y));
  return y;
#endif
}

/// @brief Calculate square root reciprocal.
DALI_HOST_DEV inline double rsqrt(double x) {
  return 1.0/sqrt(x);
}

/// @brief Calculate fast approximation of square root reciprocal.
DALI_HOST_DEV inline double fast_rsqrt(double x) {
#ifdef  __CUDA_ARCH__
  // Not likely to be used at device side anyway.
  return rsqrt(static_cast<float>(x));
#else
  // No inverse square root intrinsic for double.
  // Use bit-hack; faster than 1.0/sqrt(x).
  // https://en.wikipedia.org/wiki/Fast_inverse_square_root
  int64_t i;
  double x2, y;
  x2 = x * 0.5;
  y  = x;
  i  = *(const int64_t*)&y;
  i  = 0x5FE6EB50C7B537A9 - (i >> 1);
  y  = *(const double *)&i;
  // Three iterations of Newton-Raphson refinement give 11-12 significant digits
  y  = y * (1.5 - (x2 * y * y));
  y  = y * (1.5 - (x2 * y * y));
  y  = y * (1.5 - (x2 * y * y));
  return y;
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr float deg2rad(float deg) {
  const float d2r = M_PI/180;
  return deg * d2r;
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr float rad2deg(float rad) {
  const float r2d = 180/M_PI;
  return rad * r2d;
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr double deg2rad(double deg) {
  const double d2r = M_PI/180;
  return deg * d2r;
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr double rad2deg(double rad) {
  const double r2d = 180/M_PI;
  return rad * r2d;
}

/// @brief Calculates normalized sinc i.e. `sin(pi * x) / (pi * x)`
DALI_HOST_DEV DALI_FORCEINLINE
double sinc(double x) {
  x *= M_PI;
  if (std::abs(x) < 1e-8)
    return 1.0 - x * x * (1.0 / 6);  // remove singularity by using Taylor expansion
  return std::sin(x) / x;
}

/// @brief Calculates normalized sinc i.e. `sin(pi * x) / (pi * x)`
DALI_HOST_DEV DALI_FORCEINLINE
float sinc(float x) {
  x *= M_PI;
  if (std::abs(x) < 1e-5f)
    return 1.0f - x * x * (1.0f / 6);  // remove singularity by using Taylor expansion
  return std::sin(x) / x;
}

/// @brief Calculates power for integer arguments
template <typename X, typename Y>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<X>::value && std::is_integral<Y>::value,
                  decltype(std::declval<X>() * std::declval<X>())>
ipow(X x, Y y) {
  decltype(std::declval<X>() * std::declval<X>()) acc = 1;
  decltype(std::declval<X>() * std::declval<X>()) pow_acc = x;
  if (std::is_signed<Y>::value && std::make_signed_t<Y>(y) < 0) {
    return 0;
  }
  while (y > 0) {
    if (y & 1) {
      acc *= pow_acc;
    }
    pow_acc *= pow_acc;
    y = y >> 1;
  }
  return acc;
}

}  // namespace dali

#endif  // DALI_CORE_MATH_UTIL_H_
