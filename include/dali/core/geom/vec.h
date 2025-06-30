// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_GEOM_VEC_H_
#define DALI_CORE_GEOM_VEC_H_

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <iosfwd>
#include <ostream>
#include <cstdint>
#include <algorithm>
#include "dali/core/host_dev.h"
#include "dali/core/util.h"
#include "dali/core/math_util.h"
#include "dali/core/force_inline.h"

namespace dali {

template <int rows, int cols, typename T = float>
struct mat;

template <int N, typename T = float>
struct vec;

template <int N>
using ivec = vec<N, int32_t>;
template <int N>
using uvec = vec<N, uint32_t>;
template <int N>
using i8vec = vec<N, int8_t>;
template <int N>
using u8vec = vec<N, uint8_t>;
template <int N>
using i16vec = vec<N, int16_t>;
template <int N>
using u16vec = vec<N, uint16_t>;
/// Use only when you need to emphasise bit width - otherwise use ivec.
template <int N>
using i32vec = vec<N, int32_t>;
/// Use only when you need to emphasise bit width - otherwise use uvec.
template <int N>
using u32vec = vec<N, uint32_t>;
template <int N>
using i64vec = vec<N, int64_t>;
template <int N>
using u64vec = vec<N, uint64_t>;
template <int N>
using dvec = vec<N, double>;
template <int N>
using bvec = vec<N, bool>;

#define DEFINE_VEC_ALIASES(prefix)     \
  using prefix##vec1 = prefix##vec<1>; \
  using prefix##vec2 = prefix##vec<2>; \
  using prefix##vec3 = prefix##vec<3>; \
  using prefix##vec4 = prefix##vec<4>; \
  using prefix##vec8 = prefix##vec<8>; \
  using prefix##vec16 = prefix##vec<16>;

DEFINE_VEC_ALIASES(i)
DEFINE_VEC_ALIASES(i64)
DEFINE_VEC_ALIASES(i32)  // consider using ivec instead!
DEFINE_VEC_ALIASES(i16)
DEFINE_VEC_ALIASES(i8)
DEFINE_VEC_ALIASES(u)
DEFINE_VEC_ALIASES(u64)
DEFINE_VEC_ALIASES(u32)  // consider using uvec instead!
DEFINE_VEC_ALIASES(u16)
DEFINE_VEC_ALIASES(u8)
DEFINE_VEC_ALIASES(d)
DEFINE_VEC_ALIASES(b)
DEFINE_VEC_ALIASES()

template <typename T>
struct is_vec : std::false_type {};

template <int N, typename T>
struct is_vec<vec<N, T>> : std::true_type {};

template <typename T>
struct is_mat : std::false_type {};

template <int rows, int cols, typename Element>
struct is_mat<mat<rows, cols, Element>> : std::true_type {};

template <typename T>
using is_scalar = std::is_arithmetic<T>;


template <typename Arg1,
          typename Arg2,
          bool is_fp1 = std::is_floating_point<Arg1>::value,
          bool is_fp2 = std::is_floating_point<Arg2>::value>
struct promote_vec {
  static_assert(std::is_same<Arg1, Arg2>::value,
    "Implicit conversion of vectors only happens from integral to floating point types");
  using type = Arg1;
};

template <typename Arg1, typename Arg2>
struct promote_vec<Arg1, Arg2, true, false> {
  using type = Arg1;
};

template <typename Arg1, typename Arg2>
struct promote_vec<Arg1, Arg2, false, true> {
  using type = Arg2;
};

template <typename VecElement,
          typename Scalar,
          bool is_fp_vector = std::is_floating_point<VecElement>::value,
          bool is_fp_scalar = std::is_floating_point<Scalar>::value>
struct promote_vec_scalar {
  using type = VecElement;
};

template <typename VecElement,
          typename Scalar>
struct promote_vec_scalar<VecElement, Scalar, false, true> {
  using type = Scalar;
};

template <typename T, typename U>
using promote_vec_t = typename promote_vec<T, U>::type;

template <typename T, typename U>
using promote_vec_scalar_t = typename promote_vec_scalar<T, U>::type;

template <int N, typename T>
struct vec_base {
  constexpr vec_base() = default;

  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec_base(T scalar) {  // NOLINT
    for (int i = 0; i < N; i++)
      v[i] = scalar;
  }

  template <typename... Components,
            typename = std::enable_if_t<sizeof...(Components) == N>>
  DALI_HOST_DEV
  constexpr vec_base(Components... components) : v{T(components)... } {}  // NOLINT
  T v[N];
};

template <typename T>
struct vec_base<1, T> {
  union {
    T v[1];
    T x;
  };

  constexpr vec_base() = default;
  DALI_HOST_DEV
  constexpr vec_base(T x) : v{x} {}  // NOLINT
};

template <typename T>
struct vec_base<2, T> {
  union {
    T v[2];
    struct { T x, y; };
  };

  constexpr vec_base() = default;
  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec_base(T scalar) : v{scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(T x, T y) : v{x, y} {}
};

template <typename T>
struct vec_base<3, T> {
  union {
    T v[3];
    struct { T x, y, z; };
  };

  constexpr vec_base() = default;
  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec_base(T scalar) : v{scalar, scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(T x, T y, T z) : v{x, y, z} {}
};

template <typename T>
struct vec_base<4, T> {
  union {
    T v[4];
    struct { T x, y, z, w; };
  };

  constexpr vec_base() = default;
  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec_base(T scalar) : v{scalar, scalar, scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(T x, T y, T z, T w) : v{x, y, z, w} {}
};

template <int N, typename T>
struct vec : vec_base<N, T> {
  static_assert(std::is_standard_layout<T>::value,
                "Cannot create a vector ofa non-standard layout type");
  using element_t = T;
  constexpr vec() = default;
  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec(T scalar) : vec_base<N, T>(scalar) {}  // NOLINT

  template <typename... Components,
            typename = std::enable_if_t<sizeof...(Components) == N>>
  DALI_HOST_DEV
  constexpr vec(Components... components) : vec_base<N, T>(components...) {}  // NOLINT

  using vec_base<N, T>::v;

  template <typename U>
  DALI_HOST_DEV
  explicit constexpr vec(const mat<N, 1, U> &m) : vec(m.col(0).template cast<T>()) {}

  DALI_HOST_DEV
  constexpr vec(const mat<N, 1, T> &m) : vec(m.col(0).template cast<T>()) {}  // NOLINT

  template <typename U>
  DALI_HOST_DEV
  explicit constexpr vec(const vec<N, U> &v) : vec(v.template cast<T>()) {}

  DALI_HOST_DEV DALI_FORCEINLINE
  constexpr T &operator[](int i) { return v[i]; }
  DALI_HOST_DEV DALI_FORCEINLINE
  constexpr const T &operator[](int i) const { return v[i]; }

  template <typename U>
  DALI_HOST_DEV DALI_FORCEINLINE
  constexpr vec<N, U> cast() const {
    vec<N, U> ret = {};
    for (int i = 0; i < N; i++) {
      ret.v[i] = static_cast<U>(v[i]);
    }
    return ret;
  }

  DALI_HOST_DEV DALI_FORCEINLINE constexpr int size() const { return N; }

  DALI_HOST_DEV constexpr T *begin() { return &v[0]; }
  DALI_HOST_DEV constexpr const T *cbegin() const { return &v[0]; }
  DALI_HOST_DEV constexpr const T *begin() const { return &v[0]; }
  DALI_HOST_DEV constexpr T *end() { return &v[N]; }
  DALI_HOST_DEV constexpr const T *cend() const { return &v[N]; }
  DALI_HOST_DEV constexpr const T *end() const { return &v[N]; }

  /// @brief Calculates the sum of squares of components.
  DALI_HOST_DEV DALI_FORCEINLINE constexpr auto length_square() const {
    decltype(v[0]*v[0] + v[0]*v[0]) ret = v[0]*v[0];
    for (int i = 1; i < N; i++)
      ret += v[i]*v[i];
    return ret;
  }

  /// @brief Calculates Euclidean length of the vector.
  DALI_HOST_DEV inline auto length() const {
#ifdef __CUDA_ARCH__
    return sqrtf(length_square());
#else
    return std::sqrt(length_square());
#endif
  }
  DALI_HOST_DEV inline vec normalized() const {
    auto lsq = length_square();
    return *this * rsqrt(lsq);
  }

  /// @brief Returns a copy. Doesn't promote type to int.
  DALI_HOST_DEV DALI_FORCEINLINE constexpr vec operator+() const { return *this; }

  /// @brief Negates all components. Doesn't promote type to int.
  DALI_HOST_DEV
  DALI_FORCEINLINE constexpr vec operator-() const {
    vec<N, T> ret{};
    for (int i = 0; i < N; i++) {
      ret.v[i] = -v[i];
    }
    return ret;
  }
  DALI_HOST_DEV
  DALI_FORCEINLINE constexpr vec operator~() const {
    vec<N, T> ret{};
    for (int i = 0; i < N; i++) {
      ret.v[i] = ~v[i];
    }
    return ret;
  }

#define DEFINE_ASSIGN_VEC_OP(op)                                            \
  template <typename U>                                                     \
  DALI_HOST_DEV DALI_FORCEINLINE vec &operator op(const vec<N, U> &rhs) {   \
    for (int i = 0; i < N; i++) v[i] op rhs[i];                             \
    return *this;                                                           \
  }                                                                         \
  template <typename U>                                                     \
  DALI_HOST_DEV DALI_FORCEINLINE                                            \
  std::enable_if_t<is_scalar<U>::value, vec &> operator op(const U &rhs) {  \
    for (int i = 0; i < N; i++) v[i] op rhs;                                \
    return *this;                                                           \
  }

  DEFINE_ASSIGN_VEC_OP(=)
  DEFINE_ASSIGN_VEC_OP(+=)
  DEFINE_ASSIGN_VEC_OP(-=)
  DEFINE_ASSIGN_VEC_OP(*=)
  DEFINE_ASSIGN_VEC_OP(/=)
  DEFINE_ASSIGN_VEC_OP(%=)
  DEFINE_ASSIGN_VEC_OP(&=)
  DEFINE_ASSIGN_VEC_OP(|=)
  DEFINE_ASSIGN_VEC_OP(^=)
  DEFINE_ASSIGN_VEC_OP(<<=)
  DEFINE_ASSIGN_VEC_OP(>>=)
  #undef DEFINE_ASSIGN_VEC_OP
};


template <int N, typename T, typename U>
DALI_HOST_DEV
DALI_FORCEINLINE constexpr auto dot(const vec<N, T> &a, const vec<N, U> &b) {
  decltype(a[0]*b[0] + a[0]*b[0]) ret = a[0]*b[0];
  for (int i = 1; i < N; i++)
    ret += a[i]*b[i];
  return ret;
}

template <typename T, typename U>
DALI_HOST_DEV
DALI_FORCEINLINE constexpr auto cross(const vec<3, T> &a, const vec<3, U> &b) {
  using R = decltype(a[0]*b[0] + a[0]*b[0]);
  return vec<3, R>{
    a.y * b.z - b.y * a.z,
    a.z * b.x - b.z * a.x,
    a.x * b.y - b.x * a.y
  };
}

/// @brief Calculates `z` coordinate of a cross product of two 2D vectors
template <typename T, typename U>
DALI_HOST_DEV
DALI_FORCEINLINE constexpr auto cross(const vec<2, T> &a, const vec<2, U> &b) {
  return a.x * b.y - b.x * a.y;
}

#define DEFINE_ELEMENTIWSE_VEC_BIN_OP(op)                                            \
  template <int N, typename T, typename U>                                           \
  DALI_HOST_DEV DALI_FORCEINLINE                                                     \
  constexpr auto operator op(const vec<N, T> &a, const vec<N, U> &b) {               \
    vec<N, promote_vec_t<T, U>> ret{};                                               \
    for (int i = 0; i < N; i++) ret[i] = a[i] op b[i];                               \
    return ret;                                                                      \
  }                                                                                  \
  template <int N, typename T, typename U, typename R = promote_vec_scalar_t<T, U>>  \
  DALI_HOST_DEV DALI_FORCEINLINE                                                     \
  constexpr std::enable_if_t<is_scalar<U>::value, vec<N, R>> operator op(            \
      const vec<N, T> &a, const U &b) {                                              \
    vec<N, R> ret{};                                                                 \
    for (int i = 0; i < N; i++) ret[i] = a[i] op b;                                  \
    return ret;                                                                      \
  }                                                                                  \
  template <int N, typename T, typename U, typename R = promote_vec_scalar_t<U, T>>  \
  DALI_HOST_DEV DALI_FORCEINLINE                                                     \
  constexpr std::enable_if_t<is_scalar<T>::value, vec<N, R>> operator op(            \
      const T &a, const vec<N, U> &b) {                                              \
    vec<N, R> ret{};                                                                 \
    for (int i = 0; i < N; i++) ret[i] = a op b[i];                                  \
    return ret;                                                                      \
  }

DEFINE_ELEMENTIWSE_VEC_BIN_OP(+)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(-)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(*)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(/)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(%)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(&)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(|)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(^)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(<)  // NOLINT
DEFINE_ELEMENTIWSE_VEC_BIN_OP(>)  // NOLINT
DEFINE_ELEMENTIWSE_VEC_BIN_OP(<=)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(>=)

#define DEFINE_SHIFT_VEC_BIN_OP(op)                            \
  template <int N, typename T, typename U>                     \
  DALI_HOST_DEV DALI_FORCEINLINE constexpr vec<N, T> operator  \
  op(const vec<N, T> &a, const vec<N, U> &b) {                 \
    vec<N, T> ret{};                                           \
    for (int i = 0; i < N; i++) ret[i] = a[i] op b[i];         \
    return ret;                                                \
  }                                                            \
  template <int N, typename T, typename U>                     \
  DALI_HOST_DEV DALI_FORCEINLINE                               \
  constexpr std::enable_if_t<is_scalar<U>::value, vec<N, T>>   \
  operator op(const vec<N, T> &a, const U &b) {                \
    vec<N, T> ret{};                                           \
    for (int i = 0; i < N; i++) ret[i] = a[i] op b;            \
    return ret;                                                \
  }                                                            \
  template <int N, typename T, typename U>                     \
  DALI_HOST_DEV DALI_FORCEINLINE                               \
  constexpr std::enable_if_t<is_scalar<T>::value, vec<N, T>>   \
  operator op(const T &a, const vec<N, U> &b) {                \
    vec<N, T> ret{};                                           \
    for (int i = 0; i < N; i++) ret[i] = a op b[i];            \
    return ret;                                                \
  }

DEFINE_SHIFT_VEC_BIN_OP(<<)
DEFINE_SHIFT_VEC_BIN_OP(>>)

struct is_true {
  template <typename T>
  DALI_HOST_DEV constexpr bool operator()(const T &x) {
    return static_cast<bool>(x);
  }
};

template <int N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool all_coords(const vec<N, T> &a, Pred P = {}) {
  for (int i = 0; i < N; i++)
    if (!P(a[i]))
      return false;
  return true;
}

template <int N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool any_coord(const vec<N, T> &a, Pred P = {}) {
  for (int i = 0; i < N; i++)
    if (P(a[i]))
      return true;
  return false;
}

template <int N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator==(const vec<N, T> &a, const vec<N, U> &b) {
  for (int i = 0; i < N; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

template <int N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator!=(const vec<N, T> &a, const vec<N, U> &b) {
  for (int i = 0; i < N; i++)
    if (a[i] != b[i])
      return true;
  return false;
}

/// @brief Implements an elementwise vector function by evaluating given expression
///        with an index `i` ranging from 0 to N-1. `N` must be a compile-time constant
///        present at evaluation site.
#define IMPL_VEC_ELEMENTWISE(...)                                             \
  int i = 0;                                                               \
  using R = std::remove_cv_t<std::remove_reference_t<decltype(__VA_ARGS__)>>; \
  vec<N, R> result = {};                                                      \
  for (i = 0; i < N; i++) {                                                   \
    result[i] = (__VA_ARGS__);                                                \
  }                                                                           \
  return result;

template <typename To, int N, typename From>
DALI_HOST_DEV DALI_FORCEINLINE constexpr vec<N, To> cast(const vec<N, From> &v) {
  return v.template cast<To>();
}

template <int N, typename T>
DALI_HOST_DEV constexpr vec<N, T>
clamp(const vec<N, T> &in, const vec<N, T> &lo, const vec<N, T> &hi) {
  IMPL_VEC_ELEMENTWISE(clamp(in[i], lo[i], hi[i]));
}

#if (defined(__NVCC__) && defined(__CUDA_ARCH__)) || (defined(__clang__) && defined(__CUDA__))
template <int N>
constexpr DALI_DEVICE vec<N> floor(const vec<N> &a) {
  IMPL_VEC_ELEMENTWISE(floorf(a[i]));
}

template <int N>
constexpr DALI_DEVICE vec<N> ceil(const vec<N> &a) {
  IMPL_VEC_ELEMENTWISE(ceilf(a[i]));
}

template <int N, typename T>
constexpr DALI_DEVICE vec<N, T> min(const vec<N, T> &a, const vec<N, T> &b) {
  IMPL_VEC_ELEMENTWISE(::min(a[i], b[i]));
}

template <int N, typename T>
constexpr DALI_DEVICE vec<N, T> max(const vec<N, T> &a, const vec<N, T> &b) {
  IMPL_VEC_ELEMENTWISE(::max(a[i], b[i]));
}

#endif

#if !defined(__CUDA_ARCH__) || (defined(__clang__) && defined(__CUDA__))

template <int N, typename T>
constexpr DALI_HOST vec<N, T> floor(const vec<N, T> &a) {
  IMPL_VEC_ELEMENTWISE(std::floor(a[i]));
}

template <int N, typename T>
constexpr DALI_HOST vec<N, T> ceil(const vec<N, T> &a) {
  IMPL_VEC_ELEMENTWISE(std::ceil(a[i]));
}

template <int N, typename T>
constexpr DALI_HOST vec<N, T> min(const vec<N, T> &a, const vec<N, T> &b) {
  IMPL_VEC_ELEMENTWISE(std::min(a[i], b[i]));
}

template <int N, typename T>
constexpr DALI_HOST vec<N, T> max(const vec<N, T> &a, const vec<N, T> &b) {
  IMPL_VEC_ELEMENTWISE(std::max(a[i], b[i]));
}

#endif

template <int N>
DALI_HOST_DEV ivec<N> round_int(const vec<N> &a) {
  IMPL_VEC_ELEMENTWISE(round_int(a[i]));
}

template <int N>
DALI_HOST_DEV ivec<N> floor_int(const vec<N> &a) {
  IMPL_VEC_ELEMENTWISE(floor_int(a[i]));
}

template <int N>
DALI_HOST_DEV ivec<N> ceil_int(const vec<N> &a) {
  IMPL_VEC_ELEMENTWISE(ceil_int(a[i]));
}

template <int N, typename T, typename U>
DALI_HOST_DEV constexpr
std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                 vec<N, decltype(div_ceil(T{}, U{}))>>
div_ceil(const vec<N, T> &a, const vec<N, U> &b) {
  IMPL_VEC_ELEMENTWISE(div_ceil(a[i], b[i]));
}

template <typename T, int size0, int size1>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr auto cat(const vec<size0, T> &v0, const vec<size1, T> &v1) {
  vec<size0 + size1, T> ret = {};
  for (int i = 0; i < size0; i ++) {
    ret[i] = v0[i];
  }
  for (int i = 0; i < size1; i ++) {
    ret[i + size0] = v1[i];
  }
  return ret;
}

template <typename T, int size0>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr auto cat(const vec<size0, T> &v0, T v1) {
  vec<size0 + 1, T> ret = {};
  for (int i = 0; i < size0; i ++) {
    ret[i] = v0[i];
  }
  ret[size0] = v1;
  return ret;
}

template <typename T, int size1>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr auto cat(T v0, const vec<size1, T> &v1) {
  vec<size1 + 1, T> ret = {};
  ret[0] = v0;
  for (int i = 0; i < size1; i ++) {
    ret[i+1] = v1[i];
  }
  return ret;
}

template <typename T, int size0, int... sizes>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr auto cat(const vec<size0, T> &v0, const vec<sizes, T> &...tail) {
  return cat(v0, cat(tail...));
}

template <int sub_n, int n, typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr auto sub(const vec<n, T> &orig, int start = 0) {
  static_assert(sub_n <= n, "Cannot extend a vector using `sub` function.");
  vec<sub_n, T> ret = {};
  for (int i = 0; i < sub_n; i++)
    ret[i] = orig[i + start];
  return ret;
}

template <int... indices, int N, typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr vec<sizeof...(indices), T> shuffle(const vec<N, T> &v) {
  static_assert(all_of<(indices < N)...>::value, "Vector component index out of range");
  return { v[indices]... };
}

static_assert(is_pod_v<vec<1>>, "vec<1, T> must be a POD type");
static_assert(is_pod_v<vec<2>>, "vec<2, T> must be a POD type");
static_assert(is_pod_v<vec<3>>, "vec<3, T> must be a POD type");
static_assert(is_pod_v<vec<4>>, "vec<4, T> must be a POD type");
static_assert(is_pod_v<vec<5>>, "vec<N, T> must be a POD type");

template <int N, typename T>
std::ostream &operator<<(std::ostream& os, const dali::vec<N, T> &v) {
  for (int i = 0; i < N; i++)
    os << (i ? ", " : "{") << v[i];
  os << "}";
  return os;
}

}  // namespace dali

#ifdef __CUDACC__
template <typename T, int N>
__device__ DALI_FORCEINLINE dali::vec<N, T> __ldg(const dali::vec<N, T>* ptr) {
  using dali::vec;
  IMPL_VEC_ELEMENTWISE(__ldg(&ptr->v[i]));
}
#endif

#endif  // DALI_CORE_GEOM_VEC_H_
