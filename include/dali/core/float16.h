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
#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"
#ifndef __CUDA_ARCH__
#include "dali/util/half.hpp"
#endif

namespace dali {

struct float16 {
  DALI_HOST_DEV
  float16() = default;
  float16(const float16 &) = default;

#ifdef __CUDA_ARCH__
  __device__ float16(__half x)          : impl(x) {}  // NOLINT
  __device__ float16(float x)           : impl(x) {}  // NOLINT
  __device__ float16(double x)          : impl(x) {}  // NOLINT
  __device__ float16(signed char x)     : impl(x) {}  // NOLINT
  __device__ float16(unsigned char x)   : impl(x) {}  // NOLINT
  __device__ float16(short x)           : impl(x) {}  // NOLINT
  __device__ float16(unsigned short x)  : impl(x) {}  // NOLINT
  __device__ float16(int x)             : impl(x) {}  // NOLINT
  __device__ float16(unsigned int x)    : impl(x) {}  // NOLINT
  __device__ float16(long x)            : impl(static_cast<long long>(x)) {}  // NOLINT
  __device__ float16(unsigned long x)   : impl(static_cast<unsigned long long>(x)) {}  // NOLINT
  __device__ float16(long long x)           : impl(x) {}  // NOLINT
  __device__ float16(unsigned long long x)  : impl(x) {}  // NOLINT
#endif
#ifndef __CUDA_ARCH__
  __host__ float16(half_float::half x) : impl(x) {}  // NOLINT
  __host__ float16(float x)           : impl(x) {}  // NOLINT
  __host__ float16(double x)          : impl(x) {}  // NOLINT
  __host__ float16(signed char x)     : impl(x) {}  // NOLINT
  __host__ float16(unsigned char x)   : impl(x) {}  // NOLINT
  __host__ float16(short x)           : impl(x) {}  // NOLINT
  __host__ float16(unsigned short x)  : impl(x) {}  // NOLINT
  __host__ float16(int x)             : impl(x) {}  // NOLINT
  __host__ float16(unsigned int x)    : impl(x) {}  // NOLINT
  __host__ float16(long x)            : impl(static_cast<int64_t>(x)) {}  // NOLINT
  __host__ float16(unsigned long x)   : impl(static_cast<uint64_t>(x)) {}  // NOLINT
  __host__ float16(long long x)          : impl(static_cast<int64_t>(x)) {}  // NOLINT
  __host__ float16(unsigned long long x) : impl(static_cast<uint64_t>(x)) {}  // NOLINT
#endif


#ifdef __CUDA_ARCH__
  using impl_t = __half;
#else
  using impl_t = half_float::half;
#endif

  impl_t impl;

  DALI_HOST_DEV DALI_FORCEINLINE
  constexpr operator impl_t() const noexcept { return impl; }

#ifdef __CUDA_ARCH__
  DALI_FORCEINLINE
  __device__ operator float() const noexcept { return impl; }
#else
  DALI_FORCEINLINE
  __host__ operator float() const noexcept { return impl; }
#endif

  DALI_FORCEINLINE DALI_HOST_DEV float16 operator-() const noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
    return __hneg(impl);
#else
  return float16(-static_cast<float>(impl));
#endif
#else
    return float16(-impl);
#endif
  }

  DALI_FORCEINLINE DALI_HOST_DEV float16 operator+() const noexcept {
    return *this;
  }

  DALI_FORCEINLINE DALI_HOST_DEV float16 &operator++() noexcept;
  DALI_FORCEINLINE DALI_HOST_DEV float16 operator++(int) noexcept;

#define DALI_HALF_DECLARE_COMPOUND_ASSIGNMENT(op) \
  DALI_FORCEINLINE DALI_HOST_DEV float16 &operator op(float16 other) noexcept; \
  DALI_FORCEINLINE DALI_HOST_DEV float16 &operator op(float other) noexcept; \
  DALI_FORCEINLINE DALI_HOST_DEV float16 &operator op(double other) noexcept; \
  template <typename T> \
  DALI_FORCEINLINE DALI_HOST_DEV \
  std::enable_if_t<std::is_integral<T>::value, float16 &> \
  operator op(T other) noexcept { \
    return *this op float16(other); \
  }

  DALI_HALF_DECLARE_COMPOUND_ASSIGNMENT(+=)
  DALI_HALF_DECLARE_COMPOUND_ASSIGNMENT(-=)
  DALI_HALF_DECLARE_COMPOUND_ASSIGNMENT(*=)
  DALI_HALF_DECLARE_COMPOUND_ASSIGNMENT(/=)
};

DALI_HOST_DEV DALI_FORCEINLINE
float16 operator +(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return float16(__hadd(a.impl, b.impl));
#else
  return float16(static_cast<float>(a.impl) + static_cast<float>(b.impl));
#endif
#else
  return float16(float16::impl_t(a.impl + b.impl));
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
float16 operator -(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return float16(__hsub(a.impl, b.impl));
#else
  return float16(static_cast<float>(a.impl) - static_cast<float>(b.impl));
  return {};
#endif
#else
  return float16(float16::impl_t(a.impl - b.impl));
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
float16 operator *(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return float16(__hmul(a.impl, b.impl));
#else
  return float16(static_cast<float>(a.impl) * static_cast<float>(b.impl));
#endif
#else
  return float16(float16::impl_t(a.impl * b.impl));
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
float16 operator /(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return float16(__hdiv(a.impl, b.impl));
#else
  return float16(static_cast<float>(a.impl) / static_cast<float>(b.impl));
#endif
#else
  return float16(float16::impl_t(a.impl / b.impl));
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator ==(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __heq(a.impl, b.impl);
#else
  return static_cast<float>(a.impl) == static_cast<float>(b.impl);
#endif
#else
  return a.impl == b.impl;
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator !=(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hne(a.impl, b.impl);
#else
  return static_cast<float>(a.impl) != static_cast<float>(b.impl);
#endif
#else
  return a.impl != b.impl;
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator <(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hlt(a.impl, b.impl);
#else
  return static_cast<float>(a.impl) < static_cast<float>(b.impl);
#endif
#else
  return a.impl < b.impl;
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator <=(float16 a, float16 b) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hle(a.impl, b.impl);
#else
  return static_cast<float>(a.impl) <= static_cast<float>(b.impl);
#endif
#else
  return a.impl <= b.impl;
#endif
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator >(float16 a, float16 b) noexcept {
  return b < a;
}

DALI_HOST_DEV DALI_FORCEINLINE
bool operator >=(float16 a, float16 b) noexcept {
  return b <= a;
}

#define DALI_IMPL_HALF_FLOAT_OP(op) \
DALI_HOST_DEV DALI_FORCEINLINE \
auto operator op(float a, float16 b) noexcept { \
  return a op static_cast<float>(b); \
} \
DALI_HOST_DEV DALI_FORCEINLINE \
auto operator op(float16 a, float b) noexcept { \
  return static_cast<float>(a) op b; \
}

DALI_IMPL_HALF_FLOAT_OP(+)
DALI_IMPL_HALF_FLOAT_OP(-)
DALI_IMPL_HALF_FLOAT_OP(*)
DALI_IMPL_HALF_FLOAT_OP(/)

#define DALI_IMPL_HALF_CMP_OP(op) \
template <typename FP16, typename T> \
DALI_HOST_DEV DALI_FORCEINLINE \
std::enable_if_t<std::is_same<FP16, float16>::value, bool> \
operator op(T a, const FP16 &b) noexcept { \
  return a op static_cast<float>(b); \
} \
template <typename FP16, typename T> \
DALI_HOST_DEV DALI_FORCEINLINE \
std::enable_if_t<std::is_same<FP16, float16>::value, bool> \
operator op(const FP16 &a, T b) noexcept { \
  return static_cast<float>(a) op b; \
}

DALI_IMPL_HALF_CMP_OP(==)
DALI_IMPL_HALF_CMP_OP(!=)
DALI_IMPL_HALF_CMP_OP(<)  // NOLINT
DALI_IMPL_HALF_CMP_OP(>)  // NOLINT
DALI_IMPL_HALF_CMP_OP(<=) // NOLINT
DALI_IMPL_HALF_CMP_OP(>=) // NOLINT

DALI_HOST_DEV DALI_FORCEINLINE
float16 fma(float16 a, float16 b, float16 c) noexcept {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hfma(a.impl, b.impl, c.impl);
#else
  return float16(fmaf(static_cast<float>(a.impl),
                      static_cast<float>(b.impl),
                      static_cast<float>(c.impl)));
#endif
#else
  return static_cast<float>(a) * static_cast<float>(b) + static_cast<float>(c);
#endif
}

DALI_FORCEINLINE DALI_HOST_DEV float16 &float16::operator++() noexcept {
  *this += float16(1);
  return *this;
}

DALI_FORCEINLINE DALI_HOST_DEV float16 float16::operator++(int) noexcept {  // NOLINT misfired cast warning
  auto old = *this;
  *this += float16(1);
  return old;
}

#define DALI_HALF_FLOAT_COMPOUND_OP(op) \
DALI_FORCEINLINE DALI_HOST_DEV float16 &float16::operator op##=(float16 other) noexcept { \
  return *this = *this op other; \
} \
DALI_FORCEINLINE DALI_HOST_DEV float16 &float16::operator op##=(float other) noexcept { \
  return *this = static_cast<float>(impl) op other; \
} \
DALI_FORCEINLINE DALI_HOST_DEV float16 &float16::operator op##=(double other) noexcept { \
  return *this = static_cast<float>(impl) op other; \
}

DALI_HALF_FLOAT_COMPOUND_OP(+)
DALI_HALF_FLOAT_COMPOUND_OP(-)
DALI_HALF_FLOAT_COMPOUND_OP(*)
DALI_HALF_FLOAT_COMPOUND_OP(/)

namespace detail {

template <typename T>
struct is_half : std::false_type {};

template <>
struct is_half<float16> : std::true_type {};

#ifdef __CUDA_ARCH__
template <>
struct is_half<__half> : std::true_type {};
#else
template <>
struct is_half<half_float::half> : std::true_type {};
#endif

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

#ifdef __CUDA_ARCH__
inline __device__ dali::float16 __ldg(const dali::float16 *mem) {
  return dali::float16(__ldg(reinterpret_cast<const __half *>(mem)));
}
#endif

#endif  // DALI_CORE_FLOAT16_H_
