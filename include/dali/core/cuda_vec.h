// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CUDA_VEC_H_
#define DALI_CORE_CUDA_VEC_H_

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "dali/core/cuda_utils.h"
#include "dali/core/host_dev.h"
#include "dali/core/math_util.h"

namespace dali {

template <typename T, int N>
struct cuda_vec_type;

template <typename T, int N>
using cuda_vec_t = typename cuda_vec_type<T, N>::type;

template <typename CudaVecType>
struct cuda_vec_traits;

#define DECLARE_VEC_TRAITS(CudaVecType, ElementType, NumElements) \
  template <>                                                     \
  struct cuda_vec_traits<CudaVecType> {                           \
    using element_type = ElementType;                             \
    static constexpr int num_elements = NumElements;              \
  }

#define DECLARE_CUDA_VEC_TYPE(T, N, CUDA_VEC_T) \
  template <>                                           \
  struct cuda_vec_type<T, N> {                          \
    using type = CUDA_VEC_T;                            \
  };                                                    \
  DECLARE_VEC_TRAITS(CUDA_VEC_T, T, N)

#define DECLARE_CUDA_VEC_TYPES(T, CUDA_T) \
  DECLARE_CUDA_VEC_TYPE(T, 1, CUDA_T##1);    \
  DECLARE_CUDA_VEC_TYPE(T, 2, CUDA_T##2);    \
  DECLARE_CUDA_VEC_TYPE(T, 3, CUDA_T##3);    \
  DECLARE_CUDA_VEC_TYPE(T, 4, CUDA_T##4)


DECLARE_CUDA_VEC_TYPES(float, float);
DECLARE_CUDA_VEC_TYPES(int8_t, char);
DECLARE_CUDA_VEC_TYPES(uint8_t, uchar);
DECLARE_CUDA_VEC_TYPES(int16_t, short);  // NOLINT
DECLARE_CUDA_VEC_TYPES(uint16_t, ushort);
DECLARE_CUDA_VEC_TYPES(int32_t, int);
DECLARE_CUDA_VEC_TYPES(uint32_t, uint);

DECLARE_CUDA_VEC_TYPE(double, 1, double1);
DECLARE_CUDA_VEC_TYPE(double, 2, double2);
DECLARE_CUDA_VEC_TYPE(double, 3, double3);
DECLARE_CUDA_VEC_TYPE(double, 4, double4_16a);
DECLARE_CUDA_VEC_TYPE(int64_t, 1, long1);
DECLARE_CUDA_VEC_TYPE(int64_t, 2, long2);
DECLARE_CUDA_VEC_TYPE(int64_t, 3, long3);
DECLARE_CUDA_VEC_TYPE(int64_t, 4, long4_16a);
DECLARE_CUDA_VEC_TYPE(uint64_t, 1, ulong1);
DECLARE_CUDA_VEC_TYPE(uint64_t, 2, ulong2);
DECLARE_CUDA_VEC_TYPE(uint64_t, 3, ulong3);
DECLARE_CUDA_VEC_TYPE(uint64_t, 4, ulong4_16a);

#define DEFINE_CUDA_VEC_UNARY_OP(OP)                                              \
  template <typename CudaVecType, typename Traits = cuda_vec_traits<CudaVecType>> \
  DALI_HOST_DEV DALI_FORCEINLINE CudaVecType operator OP(const CudaVecType &a) {  \
    CudaVecType result;                                                           \
    result.x = OP a.x;                                                            \
    if constexpr (Traits::num_elements > 1)                                       \
      result.y = OP a.y;                                                          \
    if constexpr (Traits::num_elements > 2)                                       \
      result.z = OP a.z;                                                          \
    if constexpr (Traits::num_elements > 3)                                       \
      result.w = OP a.w;                                                          \
    return result;                                                                \
  }

DEFINE_CUDA_VEC_UNARY_OP(-)
DEFINE_CUDA_VEC_UNARY_OP(~)
DEFINE_CUDA_VEC_UNARY_OP(!)
DEFINE_CUDA_VEC_UNARY_OP(+)

#define DEFINE_CUDA_VEC_BINARY_OP(OP)                                             \
  template <typename CudaVecType, typename Traits = cuda_vec_traits<CudaVecType>> \
  DALI_HOST_DEV DALI_FORCEINLINE CudaVecType operator OP(const CudaVecType &a,    \
                                                         const CudaVecType &b) {  \
    CudaVecType result;                                                           \
    result.x = a.x OP b.x;                                                        \
    if constexpr (Traits::num_elements > 1)                                       \
      result.y = a.y OP b.y;                                                      \
    if constexpr (Traits::num_elements > 2)                                       \
      result.z = a.z OP b.z;                                                      \
    if constexpr (Traits::num_elements > 3)                                       \
      result.w = a.w OP b.w;                                                      \
    return result;                                                                \
  }

DEFINE_CUDA_VEC_BINARY_OP(+)
DEFINE_CUDA_VEC_BINARY_OP(-)
DEFINE_CUDA_VEC_BINARY_OP(*)
DEFINE_CUDA_VEC_BINARY_OP(/)
DEFINE_CUDA_VEC_BINARY_OP(%)
DEFINE_CUDA_VEC_BINARY_OP(&)
DEFINE_CUDA_VEC_BINARY_OP(|)
DEFINE_CUDA_VEC_BINARY_OP(^)
DEFINE_CUDA_VEC_BINARY_OP(<<)
DEFINE_CUDA_VEC_BINARY_OP(>>)

#define DEFINE_CUDA_VEC_COMPARISON_OP(OP)                                           \
  template <typename CudaVecType, typename Traits = cuda_vec_traits<CudaVecType>>   \
  DALI_HOST_DEV DALI_FORCEINLINE cuda_vec_t<int, Traits::num_elements> operator OP( \
      const CudaVecType &a, const CudaVecType &b) {                                 \
    cuda_vec_t<int, Traits::num_elements> result;                                   \
    result.x = (a.x OP b.x);                                                        \
    if constexpr (Traits::num_elements > 1)                                         \
      result.y = (a.y OP b.y);                                                      \
    if constexpr (Traits::num_elements > 2)                                         \
      result.z = (a.z OP b.z);                                                      \
    if constexpr (Traits::num_elements > 3)                                         \
      result.w = (a.w OP b.w);                                                      \
    return result;                                                                  \
  }

#define DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(OP)                                    \
  template <typename CudaVecType, typename Traits = cuda_vec_traits<CudaVecType>> \
  DALI_HOST_DEV DALI_FORCEINLINE CudaVecType &operator OP## =                     \
      (CudaVecType & a, const CudaVecType &b) {                                   \
    a = a OP b;                                                                   \
    return a;                                                                     \
  }

DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(+)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(-)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(*)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(/)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(%)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(&)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(|)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(^)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(<<)
DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(>>)

DEFINE_CUDA_VEC_COMPARISON_OP(==)
DEFINE_CUDA_VEC_COMPARISON_OP(!=)
DEFINE_CUDA_VEC_COMPARISON_OP(<)  // NOLINT
DEFINE_CUDA_VEC_COMPARISON_OP(<=)
DEFINE_CUDA_VEC_COMPARISON_OP(>)  // NOLINT
DEFINE_CUDA_VEC_COMPARISON_OP(>=)


#define DEFINE_CUDA_UNARY_BODY_1(FUNC) \
  result.x = dali::cuda::FUNC(a.x);

#define DEFINE_CUDA_UNARY_BODY_2(FUNC) \
  result.x = dali::cuda::FUNC(a.x); \
  result.y = dali::cuda::FUNC(a.y);

#define DEFINE_CUDA_UNARY_BODY_3(FUNC) \
  result.x = dali::cuda::FUNC(a.x); \
  result.y = dali::cuda::FUNC(a.y); \
  result.z = dali::cuda::FUNC(a.z);

#define DEFINE_CUDA_UNARY_BODY_4(FUNC) \
  result.x = dali::cuda::FUNC(a.x); \
  result.y = dali::cuda::FUNC(a.y); \
  result.z = dali::cuda::FUNC(a.z); \
  result.w = dali::cuda::FUNC(a.w);


#define DEFINE_CUDA_BINARY_BODY_1(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x);

#define DEFINE_CUDA_BINARY_BODY_2(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x); \
  result.y = dali::cuda::FUNC(a.y, b.y);

#define DEFINE_CUDA_BINARY_BODY_3(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x); \
  result.y = dali::cuda::FUNC(a.y, b.y); \
  result.z = dali::cuda::FUNC(a.z, b.z);

#define DEFINE_CUDA_BINARY_BODY_4(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x); \
  result.y = dali::cuda::FUNC(a.y, b.y); \
  result.z = dali::cuda::FUNC(a.z, b.z); \
  result.w = dali::cuda::FUNC(a.w, b.w);

#define DEFINE_CUDA_TERNARY_BODY_1(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x, c.x);

#define DEFINE_CUDA_TERNARY_BODY_2(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x, c.x); \
  result.y = dali::cuda::FUNC(a.y, b.y, c.y);

#define DEFINE_CUDA_TERNARY_BODY_3(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x, c.x); \
  result.y = dali::cuda::FUNC(a.y, b.y, c.y); \
  result.z = dali::cuda::FUNC(a.z, b.z, c.z);

#define DEFINE_CUDA_TERNARY_BODY_4(FUNC) \
  result.x = dali::cuda::FUNC(a.x, b.x, c.x); \
  result.y = dali::cuda::FUNC(a.y, b.y, c.y); \
  result.z = dali::cuda::FUNC(a.z, b.z, c.z); \
  result.w = dali::cuda::FUNC(a.w, b.w, c.w);

#define DEFINE_CUDA_UNARY_FUNCTION_TN(FUNC, T, N)                               \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const cuda_vec_t<T, N> &a) {         \
    cuda_vec_t<std::remove_cvref_t<decltype(dali::cuda::FUNC(a.x))>, N> result; \
    DEFINE_CUDA_UNARY_BODY_##N(FUNC);                                           \
    return result;                                                              \
  }

#define DEFINE_CUDA_BINARY_FUNCTION_TUN(FUNC, T, U, N)                                             \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const cuda_vec_t<T, N> &a, const cuda_vec_t<U, N> &b) { \
    cuda_vec_t<std::remove_cvref_t<decltype(dali::cuda::FUNC(a.x, b.x))>, N> result;               \
    DEFINE_CUDA_BINARY_BODY_##N(FUNC);                                                             \
    return result;                                                                                 \
  }

#define DEFINE_CUDA_BINARY_FUNCTION_TN(FUNC, T, N) DEFINE_CUDA_BINARY_FUNCTION_TUN(FUNC, T, T, N)

#define DEFINE_CUDA_TERNARY_FUNCTION_TUVN(FUNC, T, U, V, N)                                      \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const cuda_vec_t<T, N> &a, const cuda_vec_t<U, N> &b, \
                                           const cuda_vec_t<V, N> &c) {                          \
    cuda_vec_t<std::remove_cvref_t<decltype(dali::cuda::FUNC(a.x, b.x, c.x))>, N> result;        \
    DEFINE_CUDA_TERNARY_BODY_##N(FUNC);                                                          \
    return result;                                                                               \
  }

#define DEFINE_CUDA_TERNARY_FUNCTION_TN(FUNC, T, N) \
  DEFINE_CUDA_TERNARY_FUNCTION_TUVN(FUNC, T, T, T, N)

#define DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, T) \
  DEFINE_CUDA_##ARITY##_FUNCTION_TN(FUNC, T, 1)    \
  DEFINE_CUDA_##ARITY##_FUNCTION_TN(FUNC, T, 2)    \
  DEFINE_CUDA_##ARITY##_FUNCTION_TN(FUNC, T, 3)    \
  DEFINE_CUDA_##ARITY##_FUNCTION_TN(FUNC, T, 4)


#define DEFINE_CUDA_VEC_SIGNED_INT_FUNCTION(ARITY, FUNC) \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, int8_t)        \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, int16_t)       \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, int32_t)       \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, int64_t)

#define DEFINE_CUDA_VEC_UNSIGNED_INT_FUNCTION(ARITY, FUNC) \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, uint8_t)         \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, uint16_t)        \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, uint32_t)        \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, uint64_t)

#define DEFINE_CUDA_VEC_FP_FUNCTION(ARITY, FUNC) \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, float) \
  DEFINE_CUDA_VEC_FUNCTION_T(ARITY, FUNC, double)

#define DEFINE_CUDA_UNARY_FP_FUNCTION(FUNC) DEFINE_CUDA_VEC_FP_FUNCTION(UNARY, FUNC)

#define DEFINE_CUDA_BINARY_FP_FUNCTION(FUNC) DEFINE_CUDA_VEC_FP_FUNCTION(BINARY, FUNC)

#define DEFINE_CUDA_BINARY_FUNCTION(FUNC)             \
  DEFINE_CUDA_VEC_SIGNED_INT_FUNCTION(BINARY, FUNC)   \
  DEFINE_CUDA_VEC_UNSIGNED_INT_FUNCTION(BINARY, FUNC) \
  DEFINE_CUDA_VEC_FP_FUNCTION(BINARY, FUNC)

#define DEFINE_CUDA_TERNARY_FUNCTION(FUNC)             \
  DEFINE_CUDA_VEC_SIGNED_INT_FUNCTION(TERNARY, FUNC)   \
  DEFINE_CUDA_VEC_UNSIGNED_INT_FUNCTION(TERNARY, FUNC) \
  DEFINE_CUDA_VEC_FP_FUNCTION(TERNARY, FUNC)


namespace cuda {
#ifdef __CUDA_ARCH__
using ::abs;
using ::floor;
using ::ceil;
using ::round;
using ::sqrt;
using ::sin;
using ::cos;
using ::tan;
using ::acos;
using ::atan;
using ::exp;
using ::exp2;
using ::log;
using ::log2;
using ::log10;
using ::pow;
using ::atan2;
using ::min;
using ::max;
#else
using std::abs;
using std::floor;
using std::ceil;
using std::round;
using std::sqrt;
using std::sin;
using std::cos;
using std::tan;
using std::acos;
using std::atan;
using std::exp;
using std::exp2;
using std::log;
using std::log2;
using std::log10;
using std::pow;
using std::atan2;
using std::min;
using std::max;
#endif

using dali::clamp;
}  // namespace cuda

DEFINE_CUDA_VEC_SIGNED_INT_FUNCTION(UNARY, abs)
DEFINE_CUDA_UNARY_FP_FUNCTION(abs)
DEFINE_CUDA_UNARY_FP_FUNCTION(floor)
DEFINE_CUDA_UNARY_FP_FUNCTION(ceil)
DEFINE_CUDA_UNARY_FP_FUNCTION(round)
DEFINE_CUDA_UNARY_FP_FUNCTION(sqrt)
DEFINE_CUDA_UNARY_FP_FUNCTION(sin)
DEFINE_CUDA_UNARY_FP_FUNCTION(cos)
DEFINE_CUDA_UNARY_FP_FUNCTION(tan)
DEFINE_CUDA_UNARY_FP_FUNCTION(acos)
DEFINE_CUDA_UNARY_FP_FUNCTION(atan)
DEFINE_CUDA_UNARY_FP_FUNCTION(exp)
DEFINE_CUDA_UNARY_FP_FUNCTION(exp2)
DEFINE_CUDA_UNARY_FP_FUNCTION(log)
DEFINE_CUDA_UNARY_FP_FUNCTION(log2)
DEFINE_CUDA_UNARY_FP_FUNCTION(log10)

DEFINE_CUDA_BINARY_FP_FUNCTION(pow)
DEFINE_CUDA_BINARY_FP_FUNCTION(atan2)

DEFINE_CUDA_BINARY_FUNCTION(min)
DEFINE_CUDA_BINARY_FUNCTION(max)
DEFINE_CUDA_TERNARY_FUNCTION(clamp)

}  // namespace dali

#endif  // DALI_CORE_CUDA_VEC_H_
