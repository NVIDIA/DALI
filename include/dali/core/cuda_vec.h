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
#include <cstdint>
#include <cmath>
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

#define DECLARE_CUDA_VEC_TYPE(T, CUDA_T, N) \
  template <>                               \
  struct cuda_vec_type<T, N> {              \
    using type = CUDA_T##N;                 \
  };                                        \
  DECLARE_VEC_TRAITS(CUDA_T##N, T, N)

#define DECLARE_CUDA_VEC_TYPES(T, CUDA_T) \
  DECLARE_CUDA_VEC_TYPE(T, CUDA_T, 1);    \
  DECLARE_CUDA_VEC_TYPE(T, CUDA_T, 2);    \
  DECLARE_CUDA_VEC_TYPE(T, CUDA_T, 3);    \
  DECLARE_CUDA_VEC_TYPE(T, CUDA_T, 4)


DECLARE_CUDA_VEC_TYPES(float, float);
DECLARE_CUDA_VEC_TYPES(double, double);
DECLARE_CUDA_VEC_TYPES(int8_t, char);
DECLARE_CUDA_VEC_TYPES(uint8_t, uchar);
DECLARE_CUDA_VEC_TYPES(int16_t, short);
DECLARE_CUDA_VEC_TYPES(uint16_t, ushort);
DECLARE_CUDA_VEC_TYPES(int32_t, int);
DECLARE_CUDA_VEC_TYPES(uint32_t, uint);
DECLARE_CUDA_VEC_TYPES(int64_t, long);
DECLARE_CUDA_VEC_TYPES(uint64_t, ulong);

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
      const CudaVecType & a, const CudaVecType & b) {                               \
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

#define DEFINE_CUDA_VEC_COMPOUND_ASSIGN_OP(OP)                                        \
  template <typename CudaVecType, typename Traits = cuda_vec_traits<CudaVecType>>     \
  DALI_HOST_DEV DALI_FORCEINLINE CudaVecType &operator OP##=(CudaVecType & a,         \
                                                             const CudaVecType & b) { \
    a = a OP b;                                                                       \
    return a;                                                                         \
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
DEFINE_CUDA_VEC_COMPARISON_OP(<)
DEFINE_CUDA_VEC_COMPARISON_OP(<=)
DEFINE_CUDA_VEC_COMPARISON_OP(>)
DEFINE_CUDA_VEC_COMPARISON_OP(>=)

#define DEFINE_CUDA_UNARY_FUNCTION(FUNC)                                             \
  template <typename VecA, typename Traits = cuda_vec_traits<VecA>>                  \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const VecA &a) {                          \
    cuda_vec_t<decltype(FUNC(std::declval<VecA>().x)), Traits::num_elements> result; \
    result.x = FUNC(a.x);                                                            \
    if constexpr (Traits::num_elements > 1)                                          \
      result.y = FUNC(a.y);                                                          \
    if constexpr (Traits::num_elements > 2)                                          \
      result.z = FUNC(a.z);                                                          \
    if constexpr (Traits::num_elements > 3)                                          \
      result.w = FUNC(a.w);                                                          \
    return result;                                                                   \
  }

#define DEFINE_CUDA_BINARY_FUNCTION(FUNC)                                      \
  template <typename VecA, typename VecB,                                      \
            typename = std::enable_if_t<cuda_vec_traits<VecA>::num_elements == \
                                        cuda_vec_traits<VecB>::num_elements>>  \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const VecA &a, const VecB &b) {     \
    cuda_vec_t<decltype(FUNC(std::declval<VecA>().x, std::declval<VecB>().x)), \
               cuda_vec_traits<VecA>::num_elements>                            \
        result;                                                                \
    result.x = FUNC(a.x, b.x);                                                 \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 1)                     \
      result.y = FUNC(a.y, b.y);                                               \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 2)                     \
      result.z = FUNC(a.z, b.z);                                               \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 3)                     \
      result.w = FUNC(a.w, b.w);                                               \
    return result;                                                             \
  }

#define DEFINE_CUDA_TERNARY_FUNCTION(FUNC)                                                    \
  template <typename VecA, typename VecB, typename VecC,                                      \
            typename = std::enable_if_t<                                                      \
                cuda_vec_traits<VecA>::num_elements == cuda_vec_traits<VecB>::num_elements && \
                cuda_vec_traits<VecA>::num_elements == cuda_vec_traits<VecC>::num_elements>>  \
  DALI_HOST_DEV DALI_FORCEINLINE auto FUNC(const VecA &a, const VecB &b, const VecC &c) {     \
    cuda_vec_t<decltype(FUNC(a.x, b.x, c.x)), \
               cuda_vec_traits<VecA>::num_elements>                                           \
        result;                                                                               \
    result.x = FUNC(a.x, b.x, c.x);                                                           \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 1)                                    \
      result.y = FUNC(a.y, b.y, c.y);                                                         \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 2)                                    \
      result.z = FUNC(a.z, b.z, c.z);                                                         \
    if constexpr (cuda_vec_traits<VecA>::num_elements > 3)                                    \
      result.w = FUNC(a.w, b.w, c.w);                                                         \
    return result;                                                                            \
  }

DEFINE_CUDA_UNARY_FUNCTION(abs)
DEFINE_CUDA_UNARY_FUNCTION(sqrt)
DEFINE_CUDA_UNARY_FUNCTION(sin)
DEFINE_CUDA_UNARY_FUNCTION(cos)
DEFINE_CUDA_UNARY_FUNCTION(tan)
DEFINE_CUDA_UNARY_FUNCTION(asin)
DEFINE_CUDA_UNARY_FUNCTION(acos)
DEFINE_CUDA_UNARY_FUNCTION(atan)
DEFINE_CUDA_UNARY_FUNCTION(exp)
DEFINE_CUDA_UNARY_FUNCTION(log)
DEFINE_CUDA_UNARY_FUNCTION(log10)


DEFINE_CUDA_BINARY_FUNCTION(min)
DEFINE_CUDA_BINARY_FUNCTION(max)
DEFINE_CUDA_BINARY_FUNCTION(pow)
DEFINE_CUDA_BINARY_FUNCTION(atan2)

DEFINE_CUDA_TERNARY_FUNCTION(clamp)

}  // namespace dali

#endif  // DALI_CORE_CUDA_VEC_H_
