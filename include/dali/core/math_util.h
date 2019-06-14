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
#include "dali/core/host_dev.h"

namespace dali {

inline DALI_HOST_DEV int floor_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rd(x);
#else
  return static_cast<int>(std::floor(x));
#endif
}

inline DALI_HOST_DEV int ceil_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_ru(x);
#else
  return static_cast<int>(std::ceil(x));
#endif
}

inline DALI_HOST_DEV int round_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rn(x);
#else
  return static_cast<int>(std::roundf(x));
#endif
}

template <typename T>
DALI_HOST_DEV constexpr T clamp(const T &value, const T &lo, const T &hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

DALI_HOST_DEV inline float rsqrt(float x) {
#ifdef __CUDA_ARCH__
  return __frsqrt_rn(x);
#elif defined __SSE__
  __m128 X = _mm_set_ss(x);
  __m128 tmp = _mm_rsqrt_ss(X);
  float y = _mm_cvtss_f32(tmp);
  return y * (1.5f - x*0.5f * y*y);
#else
  int32_t i;
  float x2, y;
  x2 = x * 0.5f;
  y  = x;
  i  = *(const int32_t*)&y;
  i  = 0x5F375A86 - (i >> 1);
  y  = *(const float *)&i;
  y  = y * (1.5f - (x2 * y * y));
  y  = y * (1.5f - (x2 * y * y));
  return y;
#endif
}

DALI_HOST_DEV inline float fast_rsqrt(float x) {
#ifdef __CUDA_ARCH__
  return __frsqrt_rn(x);
#elif defined __SSE__
  __m128 X = _mm_set_ss(x);
  __m128 tmp = _mm_rsqrt_ss(X);
  return _mm_cvtss_f32(tmp);
#else
  int32_t i;
  float x2, y;
  x2 = x * 0.5f;
  y  = x;
  i  = *(const int32_t*)&y;
  i  = 0x5F375A86 - (i >> 1);
  y  = *(const float *)&i;
  y  = y * (1.5f - (x2 * y * y));
  return y;
#endif
}

DALI_HOST_DEV inline double rsqrt(double x) {
  return 1.0/sqrt(x);
}

DALI_HOST_DEV inline double fast_rsqrt(double x) {
#ifdef  __CUDA_ARCH__
  // not likely to be used at device side anyway
  return fast_rsqrt(static_cast<float>(x));
#else
  int64_t i;
  double x2, y;
  x2 = x * 0.5;
  y  = x;
  i  = *(const int64_t*)&y;
  i  = 0x5FE6EB50C7B537A9 - (i >> 1);
  y  = *(const double *)&i;
  y  = y * (1.5 - (x2 * y * y));
  y  = y * (1.5 - (x2 * y * y));
  y  = y * (1.5 - (x2 * y * y));
  return y;
#endif
}

}  // namespace dali

#endif  // DALI_CORE_MATH_UTIL_H_
