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

#ifndef DALI_CORE_GEOM_VEC_H_
#define DALI_CORE_GEOM_VEC_H_

#include <cmath>
#include "dali/core/host_dev.h"
#include "dali/core/util.h"
#include "dali/core/math_util.h"

namespace dali {

template <size_t rows, size_t cols, typename T = float>
struct mat;

template <size_t N, typename T = float>
struct vec;

template <size_t N>
using ivec = vec<N, int32_t>;
template <size_t N>
using uvec = vec<N, uint32_t>;
template <size_t N>
using i16vec = vec<N, int16_t>;
template <size_t N>
using u16vec = vec<N, uint16_t>;
template <size_t N>
using i8vec = vec<N, int8_t>;
template <size_t N>
using u8vec = vec<N, uint8_t>;
template <size_t N>
using dvec = vec<N, double>;
template <size_t N>
using bvec = vec<N, bool>;

#define DEFINE_VEC_ALIASES(prefix)\
using prefix##vec1 = prefix##vec<1>;\
using prefix##vec2 = prefix##vec<2>;\
using prefix##vec3 = prefix##vec<3>;\
using prefix##vec4 = prefix##vec<4>;\
using prefix##vec8 = prefix##vec<8>;\
using prefix##vec16 = prefix##vec<16>;\

DEFINE_VEC_ALIASES(i)
DEFINE_VEC_ALIASES(i16)
DEFINE_VEC_ALIASES(i8)
DEFINE_VEC_ALIASES(u)
DEFINE_VEC_ALIASES(u16)
DEFINE_VEC_ALIASES(u8)
DEFINE_VEC_ALIASES(d)
DEFINE_VEC_ALIASES(b)
DEFINE_VEC_ALIASES()

template <typename T>
struct is_vec : std::false_type {};

template <size_t N, typename T>
struct is_vec<vec<N, T>> : std::true_type {};

template <typename T>
struct is_mat : std::false_type {};

template <size_t rows, size_t cols, typename Element>
struct is_mat<mat<rows, cols, Element>> : std::true_type {};

template <typename T>
struct is_scalar : std::integral_constant<bool, !is_mat<T>::value && !is_vec<T>::value> {};


template <size_t N, typename T>
struct vec_base {
  constexpr vec_base() = default;

  /// @brief Distributes the scalar value to all components
  DALI_HOST_DEV
  constexpr vec_base(T scalar) {  // NOLINT
    for (size_t i = 0; i < N; i++)
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
  constexpr vec_base(const T &x) : v{x} {}  // NOLINT
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
  constexpr vec_base(const T &scalar) : v{scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y) : v{x, y} {}
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
  constexpr vec_base(const T &scalar) : v{scalar, scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y, const T &z) : v{x, y, z} {}
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
  constexpr vec_base(const T &scalar) : v{scalar, scalar, scalar, scalar} {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y, const T &z, const T &w) : v{x, y, z, w} {}
};

template <size_t N, typename T>
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
  constexpr vec(mat<N, 1, U> &m) : vec(m.col(0).template cast<T>()) {}  // NOLINT

  DALI_HOST_DEV
  constexpr T &operator[](size_t i) { return v[i]; }
  DALI_HOST_DEV
  constexpr const T &operator[](size_t i) const { return v[i]; }

  template <typename U>
  DALI_HOST_DEV
  constexpr vec<N, U> cast() const {
    vec<N, U> ret = {};
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = static_cast<U>(v[i]);
    }
    return ret;
  }

  DALI_HOST_DEV constexpr size_t size() const { return N; }

  DALI_HOST_DEV constexpr T *begin() { return &v[0]; }
  DALI_HOST_DEV constexpr const T *cbegin() const { return &v[0]; }
  DALI_HOST_DEV constexpr const T *begin() const { return &v[0]; }
  DALI_HOST_DEV constexpr T *end() { return &v[N]; }
  DALI_HOST_DEV constexpr const T *cend() const { return &v[N]; }
  DALI_HOST_DEV constexpr const T *end() const { return &v[N]; }

  /// @brief Calculates the sum of squares of components.
  DALI_HOST_DEV constexpr auto length_square() const {
    decltype(v[0]*v[0] + v[0]*v[0]) ret = v[0]*v[0];
    for (size_t i = 1; i < N; i++)
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

  /// @brief Returns a copy. Doesn't promoto type to int.
  DALI_HOST_DEV constexpr vec operator+() const { return *this; }

  /// @brief Negates all components. Doesn't promote type to int.
  DALI_HOST_DEV
  inline vec operator-() const {
    vec<N, T> ret;
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = -v[i];
    }
    return ret;
  }
  DALI_HOST_DEV
  inline vec operator~() const {
    vec<N, T> ret;
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = ~v[i];
    }
    return ret;
  }

  #define DEFINE_ASSIGN_VEC_OP(op)\
  template <typename U>\
  DALI_HOST_DEV vec &operator op(const vec<N, U> &rhs) {\
    for (size_t i = 0; i < N; i++)\
      v[i] op rhs[i];\
    return *this;\
  }\
  template <typename U>\
  DALI_HOST_DEV std::enable_if_t<!is_vec<U>::value, vec &> operator op(const U &rhs) {\
    for (size_t i = 0; i < N; i++)\
      v[i] op rhs;\
    return *this;\
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


template <size_t N, typename T, typename U>
DALI_HOST_DEV
constexpr auto dot(const vec<N, T> &a, const vec<N, U> &b) {
  decltype(a[0]*b[0] + a[0]*b[0]) ret = a[0]*b[0];
  for (size_t i = 1; i < N; i++)
    ret += a[i]*b[i];
  return ret;
}

template <typename T, typename U>
DALI_HOST_DEV
constexpr auto cross(const vec<3, T> &a, const vec<3, U> &b) {
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
constexpr auto cross(const vec<2, T> &a, const vec<2, U> &b) {
  return a.x * b.y - b.x * a.y;
}

#define DEFINE_ELEMENTIWSE_VEC_BIN_OP(op)\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV inline auto operator op(const vec<N, T> &a, const vec<N, U> &b) {\
  vec<N, decltype(T() op U())> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b[i];\
  return ret;\
}\
template <size_t N, typename T, typename U, typename R = decltype(T() op U())>\
DALI_HOST_DEV inline std::enable_if_t<!is_vec<U>::value, vec<N, R>> \
operator op(const vec<N, T> &a, const U &b) {\
  vec<N, R> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b;\
  return ret;\
}\
template <size_t N, typename T, typename U, typename R = decltype(T() op U())>\
DALI_HOST_DEV inline std::enable_if_t<!is_vec<T>::value, vec<N, R>> \
operator op(const T &a, const vec<N, U> &b) {\
  vec<N, R> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a op b[i];\
  return ret;\
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

#define DEFINE_SHIFT_VEC_BIN_OP(op)\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV vec<N, T> operator op(const vec<N, T> &a, const vec<N, U> &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b[i];\
  return ret;\
}\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV std::enable_if_t<!is_vec<U>::value, vec<N, T>> \
operator op(const vec<N, T> &a, const U &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b;\
  return ret;\
}\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV std::enable_if_t<!is_vec<T>::value, vec<N, T>> \
operator op(const T &a, const vec<N, U> &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a op b[i];\
  return ret;\
}

DEFINE_SHIFT_VEC_BIN_OP(<<)
DEFINE_SHIFT_VEC_BIN_OP(>>)

struct is_true {
  template <typename T>
  DALI_HOST_DEV constexpr bool operator()(const T &x) {
    return static_cast<bool>(x);
  }
};

template <size_t N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool all(const vec<N, T> &a, Pred P = {}) {
  for (size_t i = 0; i < N; i++)
    if (!P(a[i]))
      return false;
  return true;
}

template <size_t N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool any(const vec<N, T> &a, Pred P = {}) {
  for (size_t i = 0; i < N; i++)
    if (P(a[i]))
      return true;
  return false;
}

template <size_t N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator==(const vec<N, T> &a, const vec<N, U> &b) {
  for (size_t i = 0; i < N; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

template <size_t N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator!=(const vec<N, T> &a, const vec<N, U> &b) {
  for (size_t i = 0; i < N; i++)
    if (a[i] != b[i])
      return true;
  return false;
}

template <typename F, size_t N, typename... Elements>
DALI_HOST_DEV auto elementwise(F f, const vec<N, Elements>&... vecs) {
  using R = decltype(f(vecs[0]...));
  vec<N, R> result;
  for (size_t i = 0; i < N; i++) {
    result[i] = f(vecs[i]...);
  }
  return result;
}

template <typename To, size_t N, typename From>
DALI_HOST_DEV inline vec<N, To> cast(const vec<N, From> &v) {
  return v.template cast<To>();
}

template <size_t N, typename T>
DALI_HOST_DEV vec<N, T>
clamp(const vec<N, T> &in, const vec<N, T> &lo, const vec<N, T> &hi) {
  return elementwise(clamp, in, lo, hi);
}

template <size_t N, typename T>
DALI_HOST_DEV vec<N, T> min(const vec<N, T> &a, const vec<N, T> &b) {
  return elementwise(min, a, b);
}

template <size_t N, typename T>
DALI_HOST_DEV vec<N, T> max(const vec<N, T> &a, const vec<N, T> &b) {
  return elementwise(max, a, b);
}

#ifdef __CUDA_ARCH__
template <size_t N>
__device__ vec<N> floor(const vec<N> &a, const vec<N> &b) {
  return elementwise(floorf, a);
}

template <size_t N>
__device__ vec<N> ceil(const vec<N> &a, const vec<N> &b) {
  return elementwise(ceilf, a);
}
#else

template <size_t N, typename T>
constexpr vec<N, T> floor(const vec<N, T> &a) {
  return elementwise(std::floor, a);
}

template <size_t N, typename T>
constexpr vec<N, T> ceil(const vec<N, T> &a) {
  return elementwise(std::ceil, a);
}

#endif

template <size_t N>
DALI_HOST_DEV ivec<N> round_int(const vec<N> &a) {
  return elementwise(static_cast<int(&)(float)>(round_int), a);
}

template <typename T, size_t size0, size_t size1>
DALI_HOST_DEV
constexpr auto cat(const vec<size0, T> &v0, const vec<size1, T> &v1) {
  vec<size0 + size1, T> ret = {};
  for (size_t i = 0; i < size0; i ++) {
    ret[i] = v0[i];
  }
  for (size_t i = 0; i < size1; i ++) {
    ret[i + size0] = v1[i];
  }
  return ret;
}

template <typename T, size_t size0>
DALI_HOST_DEV
constexpr auto cat(const vec<size0, T> &v0, T v1) {
  vec<size0 + 1, T> ret = {};
  for (size_t i = 0; i < size0; i ++) {
    ret[i] = v0[i];
  }
  ret[size0] = v1;
  return ret;
}

template <typename T, size_t size1>
DALI_HOST_DEV
constexpr auto cat(T v0, const vec<size1, T> &v1) {
  vec<size1 + 1, T> ret = {};
  ret[0] = v0;
  for (size_t i = 0; i < size1; i ++) {
    ret[i+1] = v1[i];
  }
  return ret;
}

template <typename T, size_t size0, size_t... sizes>
DALI_HOST_DEV
constexpr auto cat(const vec<size0, T> &v0, const vec<sizes, T> &...tail) {
  return cat(v0, cat(tail...));
}

template <size_t sub_n, size_t n, typename T>
DALI_HOST_DEV
constexpr auto sub(const vec<n, T> &orig, size_t start = 0) {
  static_assert(sub_n <= n, "Cannot extend a vector using `sub` function.");
  vec<sub_n, T> ret = {};
  for (size_t i = 0; i < sub_n; i++)
    ret[i] = orig[i + start];
  return ret;
}

template <typename To, size_t size, typename From,
          typename OutputType = vec<size*sizeof(From)/sizeof(To), To>>
DALI_HOST_DEV
constexpr OutputType bitcast(const vec<size, From> &v) {
  static_assert(sizeof(OutputType) == sizeof(vec<size, From>),
                "Cannot bitcast to a type of different size");
  return *reinterpret_cast<const OutputType*>(reinterpret_cast<const char*>(&v));
}

static_assert(std::is_pod<vec<1>>::value, "vec<1, T> must be a POD type");
static_assert(std::is_pod<vec<2>>::value, "vec<2, T> must be a POD type");
static_assert(std::is_pod<vec<3>>::value, "vec<3, T> must be a POD type");
static_assert(std::is_pod<vec<4>>::value, "vec<4, T> must be a POD type");
static_assert(std::is_pod<vec<5>>::value, "vec<N, T> must be a POD type");

}  // namespace dali

#endif  // DALI_CORE_GEOM_VEC_H_
