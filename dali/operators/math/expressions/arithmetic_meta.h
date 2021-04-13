// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_ARITHMETIC_META_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_ARITHMETIC_META_H_

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
#include "dali/operators/math/expressions/math_overloads.h"

namespace dali {

constexpr int kMaxArity = 3;

/**
 * @brief Registered arithmetic and mathematical operations
 *
 * The enum is used to provide specializations of arithm_meta with implementation
 * for every operation.
 *
 */
enum class ArithmeticOp : int {
  // Unary arithmetic ops
  plus,
  minus,
  sqrt,
  rsqrt,
  cbrt,
  exp,
  log,
  log2,
  log10,
  abs,
  fabs,
  floor,
  ceil,
  // Trigonometric functions
  sin,
  cos,
  tan,
  asin,
  acos,
  atan,
  sinh,
  cosh,
  tanh,
  asinh,
  acosh,
  atanh,
  // Binary arithmetic ops
  add,
  sub,
  mul,
  div,
  fdiv,
  mod,
  min,
  max,
  pow,
  fpow,
  atan2,
  // Binary comparisons
  eq,   // ==
  neq,  // !=
  lt,   // <
  leq,  // <=
  gt,   // >
  geq,  // >=
  bit_and,
  bit_or,
  bit_xor,
  // Ternary Ops
  clamp
};

DALI_HOST_DEV constexpr int GetOpArity(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::plus:
    case ArithmeticOp::minus:
    case ArithmeticOp::sqrt:
    case ArithmeticOp::rsqrt:
    case ArithmeticOp::cbrt:
    case ArithmeticOp::exp:
    case ArithmeticOp::log:
    case ArithmeticOp::log2:
    case ArithmeticOp::log10:
    case ArithmeticOp::abs:
    case ArithmeticOp::fabs:
    case ArithmeticOp::floor:
    case ArithmeticOp::ceil:
    case ArithmeticOp::sin:
    case ArithmeticOp::cos:
    case ArithmeticOp::tan:
    case ArithmeticOp::asin:
    case ArithmeticOp::acos:
    case ArithmeticOp::atan:
    case ArithmeticOp::sinh:
    case ArithmeticOp::cosh:
    case ArithmeticOp::tanh:
    case ArithmeticOp::asinh:
    case ArithmeticOp::acosh:
    case ArithmeticOp::atanh:
      return 1;
    case ArithmeticOp::add:
    case ArithmeticOp::sub:
    case ArithmeticOp::mul:
    case ArithmeticOp::div:
    case ArithmeticOp::fdiv:
    case ArithmeticOp::mod:
    case ArithmeticOp::min:
    case ArithmeticOp::max:
    case ArithmeticOp::pow:
    case ArithmeticOp::fpow:
    case ArithmeticOp::atan2:
    case ArithmeticOp::eq:
    case ArithmeticOp::neq:
    case ArithmeticOp::lt:
    case ArithmeticOp::leq:
    case ArithmeticOp::gt:
    case ArithmeticOp::geq:
    case ArithmeticOp::bit_and:
    case ArithmeticOp::bit_or:
    case ArithmeticOp::bit_xor:
      return 2;
    case ArithmeticOp::clamp:
      return 3;
    default:
      return -1;
  }
}

/**
 * @brief Check if op returns floating point numbers for all inputs (promiting integers to floats)
 */
DALI_HOST_DEV constexpr bool IsIntToFloatResult(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::fdiv:
    case ArithmeticOp::sqrt:
    case ArithmeticOp::rsqrt:
    case ArithmeticOp::cbrt:
    case ArithmeticOp::exp:
    case ArithmeticOp::log:
    case ArithmeticOp::log2:
    case ArithmeticOp::log10:
    case ArithmeticOp::fabs:
    case ArithmeticOp::floor:
    case ArithmeticOp::ceil:
    case ArithmeticOp::sin:
    case ArithmeticOp::cos:
    case ArithmeticOp::tan:
    case ArithmeticOp::asin:
    case ArithmeticOp::acos:
    case ArithmeticOp::atan:
    case ArithmeticOp::sinh:
    case ArithmeticOp::cosh:
    case ArithmeticOp::tanh:
    case ArithmeticOp::asinh:
    case ArithmeticOp::acosh:
    case ArithmeticOp::atanh:
    case ArithmeticOp::fpow:
    case ArithmeticOp::atan2:
      return true;
    default:
      return false;
  }
}


DALI_HOST_DEV constexpr bool IsArithmetic(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::plus:
    case ArithmeticOp::minus:
    case ArithmeticOp::sqrt:
    case ArithmeticOp::rsqrt:
    case ArithmeticOp::cbrt:
    case ArithmeticOp::exp:
    case ArithmeticOp::log:
    case ArithmeticOp::log2:
    case ArithmeticOp::log10:
    case ArithmeticOp::abs:
    case ArithmeticOp::fabs:
    case ArithmeticOp::floor:
    case ArithmeticOp::ceil:
    case ArithmeticOp::sin:
    case ArithmeticOp::cos:
    case ArithmeticOp::tan:
    case ArithmeticOp::asin:
    case ArithmeticOp::acos:
    case ArithmeticOp::atan:
    case ArithmeticOp::sinh:
    case ArithmeticOp::cosh:
    case ArithmeticOp::tanh:
    case ArithmeticOp::asinh:
    case ArithmeticOp::acosh:
    case ArithmeticOp::atanh:
    case ArithmeticOp::add:
    case ArithmeticOp::sub:
    case ArithmeticOp::mul:
    case ArithmeticOp::div:
    case ArithmeticOp::fdiv:
    case ArithmeticOp::mod:
    case ArithmeticOp::min:
    case ArithmeticOp::max:
    case ArithmeticOp::pow:
    case ArithmeticOp::fpow:
    case ArithmeticOp::atan2:
    case ArithmeticOp::clamp:
      return true;
    default:
      return false;
  }
}

DALI_HOST_DEV constexpr bool IsBitwise(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::bit_and:
    case ArithmeticOp::bit_or:
    case ArithmeticOp::bit_xor:
      return true;
    default:
      return false;
  }
}

DALI_HOST_DEV constexpr bool IsComparison(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::eq:
    case ArithmeticOp::neq:
    case ArithmeticOp::lt:
    case ArithmeticOp::leq:
    case ArithmeticOp::gt:
    case ArithmeticOp::geq:
      return true;
    default:
      return false;
  }
}


// TODO(klecki): float16
#define ARITHMETIC_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double)

/**
 * @brief Type promotion rules
 *
 * @tparam L
 * @tparam R
 *
 * The rules are:
 * T      + T      = T
 * floatX + T      = floatX           (where T is not a float)
 * floatX + floatY = float(max(X, Y))
 * intX   + intY   = int(max(X, Y))
 * uintX  + uintY  = uint(max(X, Y))
 * intX   + uintY  = int2Y            (if X <= Y)
 * intX   + uintY  = intX             (if X > Y)
 *
 * Operating on integer types never leaves the integer types, and caps at [u]int64
 *
 * @return type - a result type of binary arithmetic operation between L and R
 */
template <typename L, typename R>
struct binary_op_promotion;

template <typename T>
struct binary_op_promotion<T, T> {
  using type = T;
};

#define REGISTER_TYPE_PROMOTION(Arg1, Arg2, Result)              \
template <>                                                      \
struct binary_op_promotion<Arg1, Arg2> { using type = Result; }; \
template <>                                                      \
struct binary_op_promotion<Arg2, Arg1> { using type = Result; }


REGISTER_TYPE_PROMOTION(float,    float16, float);
REGISTER_TYPE_PROMOTION(double,   float16, double);

REGISTER_TYPE_PROMOTION(double,   float, double);

REGISTER_TYPE_PROMOTION(bool,     float16, float16);
REGISTER_TYPE_PROMOTION(int8_t,   float16, float16);
REGISTER_TYPE_PROMOTION(uint8_t,  float16, float16);
REGISTER_TYPE_PROMOTION(int16_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint16_t, float16, float16);
REGISTER_TYPE_PROMOTION(int32_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint32_t, float16, float16);
REGISTER_TYPE_PROMOTION(int64_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint64_t, float16, float16);

REGISTER_TYPE_PROMOTION(bool,     float, float);
REGISTER_TYPE_PROMOTION(int8_t,   float, float);
REGISTER_TYPE_PROMOTION(uint8_t,  float, float);
REGISTER_TYPE_PROMOTION(int16_t,  float, float);
REGISTER_TYPE_PROMOTION(uint16_t, float, float);
REGISTER_TYPE_PROMOTION(int32_t,  float, float);
REGISTER_TYPE_PROMOTION(uint32_t, float, float);
REGISTER_TYPE_PROMOTION(int64_t,  float, float);
REGISTER_TYPE_PROMOTION(uint64_t, float, float);

REGISTER_TYPE_PROMOTION(bool,     double, double);
REGISTER_TYPE_PROMOTION(int8_t,   double, double);
REGISTER_TYPE_PROMOTION(uint8_t,  double, double);
REGISTER_TYPE_PROMOTION(int16_t,  double, double);
REGISTER_TYPE_PROMOTION(uint16_t, double, double);
REGISTER_TYPE_PROMOTION(int32_t,  double, double);
REGISTER_TYPE_PROMOTION(uint32_t, double, double);
REGISTER_TYPE_PROMOTION(int64_t,  double, double);
REGISTER_TYPE_PROMOTION(uint64_t, double, double);

REGISTER_TYPE_PROMOTION(int8_t,   bool, int8_t);
REGISTER_TYPE_PROMOTION(uint8_t,  bool, uint8_t);
REGISTER_TYPE_PROMOTION(int16_t,  bool, int16_t);
REGISTER_TYPE_PROMOTION(uint16_t, bool, uint16_t);
REGISTER_TYPE_PROMOTION(int32_t,  bool, int32_t);
REGISTER_TYPE_PROMOTION(uint32_t, bool, uint32_t);
REGISTER_TYPE_PROMOTION(int64_t,  bool, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, bool, uint64_t);

REGISTER_TYPE_PROMOTION(uint8_t,  int8_t, int16_t);
REGISTER_TYPE_PROMOTION(int16_t,  int8_t, int16_t);
REGISTER_TYPE_PROMOTION(uint16_t, int8_t, int32_t);
REGISTER_TYPE_PROMOTION(int32_t,  int8_t, int32_t);
REGISTER_TYPE_PROMOTION(uint32_t, int8_t, int64_t);
REGISTER_TYPE_PROMOTION(int64_t,  int8_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, int8_t, int64_t);

REGISTER_TYPE_PROMOTION(int16_t,  uint8_t, int16_t);
REGISTER_TYPE_PROMOTION(uint16_t, uint8_t, uint16_t);
REGISTER_TYPE_PROMOTION(int32_t,  uint8_t, int32_t);
REGISTER_TYPE_PROMOTION(uint32_t, uint8_t, uint32_t);
REGISTER_TYPE_PROMOTION(int64_t,  uint8_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, uint8_t, uint64_t);

REGISTER_TYPE_PROMOTION(uint16_t, int16_t, int32_t);
REGISTER_TYPE_PROMOTION(int32_t,  int16_t, int32_t);
REGISTER_TYPE_PROMOTION(uint32_t, int16_t, int64_t);
REGISTER_TYPE_PROMOTION(int64_t,  int16_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, int16_t, int64_t);

REGISTER_TYPE_PROMOTION(int32_t,  uint16_t, int32_t);
REGISTER_TYPE_PROMOTION(uint32_t, uint16_t, uint32_t);
REGISTER_TYPE_PROMOTION(int64_t,  uint16_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, uint16_t, uint64_t);

REGISTER_TYPE_PROMOTION(uint32_t, int32_t, int64_t);
REGISTER_TYPE_PROMOTION(int64_t,  int32_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, int32_t, int64_t);

REGISTER_TYPE_PROMOTION(int64_t,  uint32_t, int64_t);
REGISTER_TYPE_PROMOTION(uint64_t, uint32_t, uint64_t);

REGISTER_TYPE_PROMOTION(uint64_t, int64_t, int64_t);

template <typename L, typename R>
using binary_result_t = typename binary_op_promotion<L, R>::type;

/**
 * @brief Struct intended as a mapping from ArithmeticOp enum to it's implemetation.
 *
 * It should provide an `impl` static member function of required arity accepting scalar inputs
 * of arbitrary arithmetic types and the template allowing to calculate it's output type
 *
 * It also contains input and output counts as well as to_string member function.
 *
 * The full specification is provided in comment below
 *
 * @tparam op  Mapped Op.
 * @tparam Backend Allows to specialize for given backend
 */
template <ArithmeticOp op, typename Backend>
struct arithm_meta;

// Specification for arithm_meta
//
// template <ArithmeticOp op, typename Backend>
// struct arithm_meta {
//   /**
//    * @brief Alias taking `num_inputs` types and returning a result type of the operation
//    * For example for regular arithmetic binary operations it should be `binary_result_t`
//    */
//    template <typename... T>
//    using result_t = ...;
//
//   /**
//    * @brief Implementation of the operation on scalar values
//    *
//    * @tparam T Types of the arguments
//    * @param t scalar values of the arguments
//    * @return result of the oparation
//    */
//    template <typename... T>
//    DALI_HOST_DEV static constexpr result_t<T...> impl(T... t);
//
//   /**
//    * @brief Simple representation of the operation, like `+` or `sin`
//    */
//    static std::string to_string();
//
//   /**
//    * @brief Constants defining the number of inputs and outputs of the operation
//    */
//    static constexpr int num_inputs;
//    static constexpr int num_outputs;
// };


/************************************************************/
/*                                                          */
/* Section for registering implementations of for AritihmOp */
/*                                                          */
/************************************************************/


#define REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, BACKEND)                        \
  template <>                                                                       \
  struct arithm_meta<OP, BACKEND> {                                                 \
    template <typename T>                                                           \
    using result_t = T;                                                             \
                                                                                    \
    template <typename T>                                                           \
    DALI_HOST_DEV static constexpr result_t<T> impl(T v) {                          \
      static_assert(GetOpArity(OP) == 1,                                            \
                    "Registered operation arity does not match the requirements."); \
      auto v_ = static_cast<result_t<T>>(v);                                        \
      return EXPRESSION v_;                                                         \
    }                                                                               \
                                                                                    \
    static inline std::string to_string() {                                         \
      return #EXPRESSION;                                                           \
    }                                                                               \
                                                                                    \
    static constexpr int num_inputs = 1;                                            \
    static constexpr int num_outputs = 1;                                           \
  }

#define REGISTER_UNARY_IMPL(OP, EXPRESSION)                \
  REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, CPUBackend); \
  REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, GPUBackend)

REGISTER_UNARY_IMPL(ArithmeticOp::plus, +);
REGISTER_UNARY_IMPL(ArithmeticOp::minus, -);


#define REGISTER_UNARY_FUNC_IMPL(MATH_FUNC)                                         \
  template <typename Backend>                                                       \
  struct arithm_meta<ArithmeticOp::MATH_FUNC, Backend> {                            \
    template <typename T>                                                           \
    using result_t = T;                                                             \
                                                                                    \
    template <typename T>                                                           \
    DALI_HOST_DEV static constexpr result_t<T> impl(T v) {                          \
      static_assert(GetOpArity(ArithmeticOp::MATH_FUNC) == 1,                       \
                    "Registered operation arity does not match the requirements."); \
      auto v_ = static_cast<result_t<T>>(v);                                        \
      return math_##MATH_FUNC(v_);                                                  \
    }                                                                               \
                                                                                    \
    static inline std::string to_string() {                                         \
      return #MATH_FUNC;                                                            \
    }                                                                               \
                                                                                    \
    static constexpr int num_inputs = 1;                                            \
    static constexpr int num_outputs = 1;                                           \
  }


REGISTER_UNARY_FUNC_IMPL(abs);


#define REGISTER_UNARY_FUNC_FLOAT_IMPL(MATH_FUNC)                            \
  template <typename Backend>                                                \
  struct arithm_meta<ArithmeticOp::MATH_FUNC, Backend> {                     \
    template <typename T>                                                    \
    using result_t = std::conditional_t<!is_fp_or_half<T>::value, float, T>; \
                                                                             \
    template <typename T>                                                    \
    DALI_HOST_DEV static constexpr result_t<T> impl(T v) {                   \
      auto v_ = static_cast<result_t<T>>(v);                                 \
      return math_##MATH_FUNC(v_);                                           \
    }                                                                        \
                                                                             \
    static inline std::string to_string() {                                  \
      return #MATH_FUNC;                                                     \
    }                                                                        \
                                                                             \
    static constexpr int num_inputs = 1;                                     \
    static constexpr int num_outputs = 1;                                    \
  };

REGISTER_UNARY_FUNC_FLOAT_IMPL(sqrt);
REGISTER_UNARY_FUNC_FLOAT_IMPL(rsqrt);
REGISTER_UNARY_FUNC_FLOAT_IMPL(cbrt);
REGISTER_UNARY_FUNC_FLOAT_IMPL(exp);
REGISTER_UNARY_FUNC_FLOAT_IMPL(log);
REGISTER_UNARY_FUNC_FLOAT_IMPL(log2);
REGISTER_UNARY_FUNC_FLOAT_IMPL(log10);
REGISTER_UNARY_FUNC_FLOAT_IMPL(fabs);
REGISTER_UNARY_FUNC_FLOAT_IMPL(floor);
REGISTER_UNARY_FUNC_FLOAT_IMPL(ceil);
REGISTER_UNARY_FUNC_FLOAT_IMPL(sin);
REGISTER_UNARY_FUNC_FLOAT_IMPL(cos);
REGISTER_UNARY_FUNC_FLOAT_IMPL(tan);
REGISTER_UNARY_FUNC_FLOAT_IMPL(asin);
REGISTER_UNARY_FUNC_FLOAT_IMPL(acos);
REGISTER_UNARY_FUNC_FLOAT_IMPL(atan);
REGISTER_UNARY_FUNC_FLOAT_IMPL(sinh);
REGISTER_UNARY_FUNC_FLOAT_IMPL(cosh);
REGISTER_UNARY_FUNC_FLOAT_IMPL(tanh);
REGISTER_UNARY_FUNC_FLOAT_IMPL(asinh);
REGISTER_UNARY_FUNC_FLOAT_IMPL(acosh);
REGISTER_UNARY_FUNC_FLOAT_IMPL(atanh);


#define REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, BACKEND)                       \
  template <>                                                                       \
  struct arithm_meta<OP, BACKEND> {                                                 \
    template <typename L, typename R>                                               \
    using result_t = binary_result_t<L, R>;                                         \
                                                                                    \
    template <typename L, typename R>                                               \
    DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {                  \
      static_assert(GetOpArity(OP) == 2,                                            \
                    "Registered operation arity does not match the requirements."); \
      auto l_ = static_cast<result_t<L, R>>(l);                                     \
      auto r_ = static_cast<result_t<L, R>>(r);                                     \
      return l_ EXPRESSION r_;                                                      \
    }                                                                               \
                                                                                    \
    static inline std::string to_string() {                                         \
      return #EXPRESSION;                                                           \
    }                                                                               \
                                                                                    \
    static constexpr int num_inputs = 2;                                            \
    static constexpr int num_outputs = 1;                                           \
  }

#define REGISTER_BINARY_IMPL(OP, EXPRESSION)                \
  REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, CPUBackend); \
  REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, GPUBackend)

REGISTER_BINARY_IMPL(ArithmeticOp::add, +);
REGISTER_BINARY_IMPL(ArithmeticOp::sub, -);
REGISTER_BINARY_IMPL(ArithmeticOp::mul, *);

template <typename Backend>
struct arithm_meta<ArithmeticOp::min, Backend> {
  template <typename L, typename R>
  using result_t = binary_result_t<L, R>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return l_ < r_ ? l_ : r_;
  }

  static inline std::string to_string() {
    return "min";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <typename Backend>
struct arithm_meta<ArithmeticOp::max, Backend> {
  template <typename L, typename R>
  using result_t = binary_result_t<L, R>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return l_ > r_ ? l_ : r_;
  }

  static inline std::string to_string() {
    return "max";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <typename Backend>
struct arithm_meta<ArithmeticOp::pow, Backend> {
  template <typename L, typename R>
  using result_t = binary_result_t<L, R>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return math_pow(l_, r_);
  }

  static inline std::string to_string() {
    return "pow";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <typename Backend>
struct arithm_meta<ArithmeticOp::fpow, Backend> {
  template <typename L, typename R>
  using result_t = std::conditional_t<!is_fp_or_half<L>::value && !is_fp_or_half<R>::value, float,
                                      binary_result_t<L, R>>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return math_pow(l_, r_);
  }

  static inline std::string to_string() {
    return "fpow";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <typename Backend>
struct arithm_meta<ArithmeticOp::atan2, Backend> {
  template <typename L, typename R>
  using result_t = std::conditional_t<!is_fp_or_half<L>::value && !is_fp_or_half<R>::value, float,
                                      binary_result_t<L, R>>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return math_atan2(l_, r_);
  }

  static inline std::string to_string() {
    return "atan2";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};


template <typename Backend>
struct arithm_meta<ArithmeticOp::clamp, Backend> {
  template <typename T, typename Min, typename Max>
  using result_t = binary_result_t<binary_result_t<T, Min>, Max>;

  template <typename T, typename Min, typename Max>
  DALI_HOST_DEV static constexpr result_t<T, Min, Max> impl(T v, Min lo, Max hi) {
    return clamp<result_t<T, Min, Max>>(v, lo, hi);
  }

  static inline std::string to_string() {
    return "clamp";
  }

  static constexpr int num_inputs = 3;
  static constexpr int num_outputs = 1;
};


// Specialization for mul and bool so we use && instead of * so the compiler is happy
template <>
DALI_HOST_DEV constexpr binary_result_t<bool, bool>
  arithm_meta<ArithmeticOp::mul, CPUBackend>::impl<bool, bool>(bool l, bool r) {
  return l && r;
}

template <>
DALI_HOST_DEV constexpr binary_result_t<bool, bool>
  arithm_meta<ArithmeticOp::mul, GPUBackend>::impl<bool, bool>(bool l, bool r) {
  return l && r;
}

REGISTER_BINARY_IMPL(ArithmeticOp::div, /);

#define REGISTER_BINARY_BITWISE_IMPL_BACKEND(OP, EXPRESSION, BACKEND)                \
  template <>                                                                        \
  struct arithm_meta<OP, BACKEND> {                                                  \
    template <typename L, typename R>                                                \
    using result_t = binary_result_t<L, R>;                                          \
                                                                                     \
    template <typename L, typename R>                                                \
    using bitwise_allowed =                                                          \
        std::enable_if_t<std::is_integral<L>::value && std::is_integral<R>::value,   \
                         result_t<L, R>>;                                            \
    template <typename L, typename R>                                                \
    using bitwise_disallowed =                                                       \
        std::enable_if_t<!std::is_integral<L>::value || !std::is_integral<R>::value, \
                         result_t<L, R>>;                                            \
                                                                                     \
    template <typename L, typename R>                                                \
    DALI_HOST_DEV static constexpr bitwise_allowed<L, R> impl(L l, R r) {            \
      static_assert(GetOpArity(OP) == 2,                                             \
                    "Registered operation arity does not match the requirements.");  \
      auto l_ = static_cast<result_t<L, R>>(l);                                      \
      auto r_ = static_cast<result_t<L, R>>(r);                                      \
      return l_ EXPRESSION r_;                                                       \
    }                                                                                \
    template <typename L, typename R>                                                \
    DALI_HOST_DEV static constexpr bitwise_disallowed<L, R> impl(L l, R r) {         \
      return {};                                                                     \
    }                                                                                \
                                                                                     \
    static inline std::string to_string() {                                          \
      return #EXPRESSION;                                                            \
    }                                                                                \
                                                                                     \
    static constexpr int num_inputs = 2;                                             \
    static constexpr int num_outputs = 1;                                            \
  }

#define REGISTER_BINARY_BITWISE_IMPL(OP, EXPRESSION)                \
  REGISTER_BINARY_BITWISE_IMPL_BACKEND(OP, EXPRESSION, CPUBackend); \
  REGISTER_BINARY_BITWISE_IMPL_BACKEND(OP, EXPRESSION, GPUBackend)



REGISTER_BINARY_BITWISE_IMPL(ArithmeticOp::bit_and, &);
REGISTER_BINARY_BITWISE_IMPL(ArithmeticOp::bit_or,  |);
REGISTER_BINARY_BITWISE_IMPL(ArithmeticOp::bit_xor, ^);

// @TODO(klecki): move it somewhere appropriate

template <typename T>
struct type_wrapper {
  using type = T;
};

template <typename T>
struct to_unsigned;

template <>
struct to_unsigned<int8_t> : type_wrapper<uint8_t>{};
template <>
struct to_unsigned<int16_t> : type_wrapper<uint16_t>{};
template <>
struct to_unsigned<int32_t> : type_wrapper<uint32_t>{};
template <>
struct to_unsigned<int64_t> : type_wrapper<uint64_t>{};

template <typename T>
using to_unsigned_t = typename to_unsigned<T>::type;


template <typename T>
struct to_signed;

template <>
struct to_signed<uint8_t> : type_wrapper<int8_t>{};
template <>
struct to_signed<uint16_t> : type_wrapper<int16_t>{};
template <>
struct to_signed<uint32_t> : type_wrapper<int32_t>{};
template <>
struct to_signed<uint64_t> : type_wrapper<int64_t>{};

template <typename T>
using to_signed_t = typename to_signed<T>::type;

/**
 * @brief Create a template class with cmp static member function for safe comparisons
 *
 * @param EXPR the comparison expression, for example `==` or `<=`
 * @param NAME the name of the operation added as suffix to `safe_compare_` class name
 * @param LEFT_NEGATIVE what should be returned if the left operand is negative
 *                      and it is compared with unsigned operand.
 * @param RIGHT_NEGATIVE as above but for right negative operand compared with left unsigned type
 *
 * For example if we have `<=`, when comparing negative value on the left with unsigned value on the
 * right, the result is always true, like `-1 <= 1u`.
 *
 * The result of cmp is direct use of the provided EXPR or special case for comparing signed
 * and unsigned integers that handles the negative case
 */
#define REGISTER_SAFE_COMPARE(EXPR, NAME, LEFT_NEGATIVE, RIGHT_NEGATIVE)                      \
  template <typename Left, typename Right,                                                    \
            bool LeftSigned = std::is_integral<Left>::value &&std::is_signed<Left>::value,    \
            bool RightSigned = std::is_integral<Right>::value &&std::is_signed<Right>::value> \
  struct safe_compare_##NAME {                                                                \
    DALI_HOST_DEV static constexpr bool cmp(Left l, Right r) {                                \
      return l EXPR r;                                                                        \
    }                                                                                         \
  };                                                                                          \
  template <typename Left, typename Right>                                                    \
  struct safe_compare_##NAME<Left, Right, true, false> {                                      \
    DALI_HOST_DEV static constexpr bool cmp(Left l, Right r) {                                \
      if (l < 0) {                                                                            \
        return LEFT_NEGATIVE;                                                                 \
      }                                                                                       \
      return static_cast<to_unsigned_t<Left>>(l) EXPR r;                                      \
    }                                                                                         \
  };                                                                                          \
  template <typename Left, typename Right>                                                    \
  struct safe_compare_##NAME<Left, Right, false, true> {                                      \
    DALI_HOST_DEV static constexpr bool cmp(Left l, Right r) {                                \
      if (r < 0) {                                                                            \
        return RIGHT_NEGATIVE;                                                                \
      }                                                                                       \
      return l EXPR static_cast<to_unsigned_t<Right>>(r);                                     \
    }                                                                                         \
  };

REGISTER_SAFE_COMPARE(==, eq,  false, false);
REGISTER_SAFE_COMPARE(!=, neq, true,  true);
REGISTER_SAFE_COMPARE(<,  lt,  true,  false);
REGISTER_SAFE_COMPARE(<=, leq, true,  false);
REGISTER_SAFE_COMPARE(>,  gt,  false, true);
REGISTER_SAFE_COMPARE(>=, geq, false, true);

#define REGISTER_COMPARISON_IMPL_BACKEND(OP, EXPRESSION, NAME, BACKEND)             \
  template <>                                                                       \
  struct arithm_meta<OP, BACKEND> {                                                 \
    template <typename L, typename R>                                               \
    using result_t = bool;                                                          \
                                                                                    \
    template <typename L, typename R>                                               \
    DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {                  \
      static_assert(GetOpArity(OP) == 2,                                            \
                    "Registered operation arity does not match the requirements."); \
      return safe_compare_##NAME<L, R>::cmp(l, r);                                  \
    }                                                                               \
                                                                                    \
    static inline std::string to_string() {                                         \
      return #EXPRESSION;                                                           \
    }                                                                               \
                                                                                    \
    static constexpr int num_inputs = 2;                                            \
    static constexpr int num_outputs = 1;                                           \
  }

#define REGISTER_COMPARISON_IMPL(OP, EXPRESSION, NAME)          \
  REGISTER_COMPARISON_IMPL_BACKEND(OP, EXPRESSION, NAME, CPUBackend); \
  REGISTER_COMPARISON_IMPL_BACKEND(OP, EXPRESSION, NAME, GPUBackend)

REGISTER_COMPARISON_IMPL(ArithmeticOp::eq,  ==, eq);
REGISTER_COMPARISON_IMPL(ArithmeticOp::neq, !=, neq);
REGISTER_COMPARISON_IMPL(ArithmeticOp::lt,  <,  lt);
REGISTER_COMPARISON_IMPL(ArithmeticOp::leq, <=, leq);
REGISTER_COMPARISON_IMPL(ArithmeticOp::gt,  >,  gt);
REGISTER_COMPARISON_IMPL(ArithmeticOp::geq, >=, geq);


template <typename Backend>
struct arithm_meta<ArithmeticOp::fdiv, Backend> {
  template <typename L, typename R>
  using result_t = std::conditional_t<!is_fp_or_half<L>::value && !is_fp_or_half<R>::value, float,
                                      binary_result_t<L, R>>;

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr result_t<L, R> impl(L l, R r) {
    auto l_ = static_cast<result_t<L, R>>(l);
    auto r_ = static_cast<result_t<L, R>>(r);
    return l_ / r_;
  }

  static inline std::string to_string() {
    return "//";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <>
struct arithm_meta<ArithmeticOp::mod, CPUBackend> {
  template <typename L, typename R>
  using result_t = binary_result_t<L, R>;

  template <typename L, typename R>
  static constexpr std::enable_if_t<
      std::is_integral<L>::value && std::is_integral<R>::value, result_t<L, R>>
  impl(L l, R r) {
    return l % r;
  }

  template <typename L, typename R>
  static constexpr std::enable_if_t<
      !std::is_integral<L>::value || !std::is_integral<R>::value, result_t<L, R>>
  impl(L l, R r) {
    using L_promotion = std::conditional_t<std::is_same<float16, L>::value, float, L>;
    using R_promotion = std::conditional_t<std::is_same<float16, R>::value, float, R>;
    return std::remainder(static_cast<L_promotion>(l), static_cast<R_promotion>(r));
  }

  static std::string to_string() {
    return "%";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};

template <>
struct arithm_meta<ArithmeticOp::mod, GPUBackend> {
  template <typename L, typename R>
  using result_t = binary_result_t<L, R>;

  template <typename L, typename R>
  __device__ static constexpr std::enable_if_t<
      std::is_integral<L>::value && std::is_integral<R>::value, result_t<L, R>>
  impl(L l, R r) {
    return l % r;
  }

  template <typename L, typename R>
  __device__ static constexpr std::enable_if_t<
      (!std::is_integral<L>::value || !std::is_integral<R>::value) &&
          (sizeof(L) < sizeof(double) && sizeof(R) < sizeof(double)),
      result_t<L, R>>
  impl(L l, R r) {
    return remainderf(static_cast<float>(l), static_cast<float>(r));
  }

  template <typename L, typename R>
  __device__ static constexpr std::enable_if_t<
      (!std::is_integral<L>::value || !std::is_integral<R>::value) &&
          (sizeof(L) >= sizeof(double) || sizeof(R) >= sizeof(double)),
      binary_result_t<L, R>>
  impl(L l, R r) {
    return remainder(static_cast<double>(l), static_cast<double>(r));
  }

  static std::string to_string() {
    return "%";
  }

  static constexpr int num_inputs = 2;
  static constexpr int num_outputs = 1;
};


inline std::string to_string(ArithmeticOp op) {
  std::string result;
  VALUE_SWITCH(op, op_static,
               (ArithmeticOp::plus, ArithmeticOp::minus,  ArithmeticOp::sqrt, ArithmeticOp::rsqrt,
                ArithmeticOp::cbrt, ArithmeticOp::exp,  ArithmeticOp::log, ArithmeticOp::log2,
                ArithmeticOp::log10, ArithmeticOp::abs, ArithmeticOp::fabs, ArithmeticOp::floor,
                ArithmeticOp::ceil,
                ArithmeticOp::sin, ArithmeticOp::cos, ArithmeticOp::tan, ArithmeticOp::asin,
                ArithmeticOp::acos, ArithmeticOp::atan, ArithmeticOp::sinh, ArithmeticOp::cosh,
                ArithmeticOp::tanh, ArithmeticOp::asinh, ArithmeticOp::acosh, ArithmeticOp::atanh,
                ArithmeticOp::add, ArithmeticOp::sub, ArithmeticOp::mul, ArithmeticOp::div,
                ArithmeticOp::fdiv, ArithmeticOp::mod, ArithmeticOp::min, ArithmeticOp::max,
                ArithmeticOp::pow, ArithmeticOp::fpow, ArithmeticOp::atan2,
                ArithmeticOp::eq, ArithmeticOp::neq, ArithmeticOp::lt, ArithmeticOp::leq,
                ArithmeticOp::gt, ArithmeticOp::geq,
                ArithmeticOp::bit_and, ArithmeticOp::bit_or, ArithmeticOp::bit_xor,
                ArithmeticOp::clamp),
               (result = arithm_meta<op_static, CPUBackend>::to_string();),
               (result = "InvalidOp";));  // NOLINT(whitespace/parens)
  return result;
}

/**
 * @brief Calculate type promotion of Binary Arithmetic op at runtime
 */
DLL_PUBLIC DALIDataType BinaryTypePromotion(DALIDataType left, DALIDataType right);
/**
 * @brief Calculate type promotion for given `op` and input `types` at runtime
 */
DLL_PUBLIC DALIDataType TypePromotion(ArithmeticOp op, span<DALIDataType> types);


inline ArithmeticOp NameToOp(const std::string &op_name) {
  static std::map<std::string, ArithmeticOp> token_to_op = {
      {"plus",   ArithmeticOp::plus},
      {"minus",  ArithmeticOp::minus},
      {"exp",    ArithmeticOp::exp},
      {"sqrt",   ArithmeticOp::sqrt},
      {"rsqrt",  ArithmeticOp::rsqrt},
      {"cbrt",   ArithmeticOp::cbrt},
      {"log",    ArithmeticOp::log},
      {"log2",   ArithmeticOp::log2},
      {"log10",  ArithmeticOp::log10},
      {"abs",    ArithmeticOp::abs},
      {"fabs",   ArithmeticOp::fabs},
      {"floor",  ArithmeticOp::floor},
      {"ceil",   ArithmeticOp::ceil},
      {"sin",    ArithmeticOp::sin},
      {"cos",    ArithmeticOp::cos},
      {"tan",    ArithmeticOp::tan},
      {"asin",   ArithmeticOp::asin},
      {"acos",   ArithmeticOp::acos},
      {"atan",   ArithmeticOp::atan},
      {"sinh",   ArithmeticOp::sinh},
      {"cosh",   ArithmeticOp::cosh},
      {"tanh",   ArithmeticOp::tanh},
      {"asinh",  ArithmeticOp::asinh},
      {"acosh",  ArithmeticOp::acosh},
      {"atanh",  ArithmeticOp::atanh},
      {"add",    ArithmeticOp::add},
      {"sub",    ArithmeticOp::sub},
      {"mul",    ArithmeticOp::mul},
      {"div",    ArithmeticOp::div},
      {"fdiv",   ArithmeticOp::fdiv},
      {"mod",    ArithmeticOp::mod},
      {"min",    ArithmeticOp::min},
      {"max",    ArithmeticOp::max},
      {"pow",    ArithmeticOp::pow},
      {"fpow",   ArithmeticOp::fpow},
      {"atan2",  ArithmeticOp::atan2},
      {"eq",     ArithmeticOp::eq},
      {"neq",    ArithmeticOp::neq},
      {"lt",     ArithmeticOp::lt},
      {"leq",    ArithmeticOp::leq},
      {"gt",     ArithmeticOp::gt},
      {"geq",    ArithmeticOp::geq},
      {"bitand", ArithmeticOp::bit_and},
      {"bitor",  ArithmeticOp::bit_or},
      {"bitxor", ArithmeticOp::bit_xor},
      {"clamp",  ArithmeticOp::clamp},
  };
  auto it = token_to_op.find(op_name);
  DALI_ENFORCE(it != token_to_op.end(), "No implementation for op \"" + op_name + "\".");
  return it->second;
}

/**
 * @brief Check if input of given `shape` should be considered to represent (tensor of) scalars.
 *
 * As a backward compatibility, 1D 1-element tensors are considered scalars (in addition to true
 * scalars).
 */
inline bool IsScalarLike(const TensorListShape<> &shape) {
  return is_uniform(shape) && shape.sample_dim() <= 1 && volume(shape.tensor_shape_span(0)) == 1;
}

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_ARITHMETIC_META_H_
