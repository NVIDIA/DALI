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

#ifndef DALI_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_
#define DALI_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "dali/core/cuda_utils.h"
#include "dali/core/small_vector.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

constexpr int kMaxArity = 2;

enum class ArithmeticOp : int {
  plus,
  minus,
  add,
  sub,
  mul,
  div,
  fdiv,
  mod,
};

DALI_HOST_DEV constexpr int GetOpArity(ArithmeticOp op) {
  switch (op) {
    case ArithmeticOp::plus:
    case ArithmeticOp::minus:
      return 1;
    case ArithmeticOp::add:
    case ArithmeticOp::sub:
    case ArithmeticOp::mul:
    case ArithmeticOp::div:
    case ArithmeticOp::fdiv:
    case ArithmeticOp::mod:
      return 2;
    default:
      return -1;
  }
}

// TODO(klecki): float16
#define ARITHMETIC_ALLOWED_TYPES \
  (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double)

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


REGISTER_TYPE_PROMOTION(float,   float16, float);
REGISTER_TYPE_PROMOTION(double,  float16, double);

REGISTER_TYPE_PROMOTION(double,  float, double);

REGISTER_TYPE_PROMOTION(int8_t,   float16, float16);
REGISTER_TYPE_PROMOTION(uint8_t,  float16, float16);
REGISTER_TYPE_PROMOTION(int16_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint16_t, float16, float16);
REGISTER_TYPE_PROMOTION(int32_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint32_t, float16, float16);
REGISTER_TYPE_PROMOTION(int64_t,  float16, float16);
REGISTER_TYPE_PROMOTION(uint64_t, float16, float16);

REGISTER_TYPE_PROMOTION(int8_t,   float, float);
REGISTER_TYPE_PROMOTION(uint8_t,  float, float);
REGISTER_TYPE_PROMOTION(int16_t,  float, float);
REGISTER_TYPE_PROMOTION(uint16_t, float, float);
REGISTER_TYPE_PROMOTION(int32_t,  float, float);
REGISTER_TYPE_PROMOTION(uint32_t, float, float);
REGISTER_TYPE_PROMOTION(int64_t,  float, float);
REGISTER_TYPE_PROMOTION(uint64_t, float, float);

REGISTER_TYPE_PROMOTION(int8_t,   double, double);
REGISTER_TYPE_PROMOTION(uint8_t,  double, double);
REGISTER_TYPE_PROMOTION(int16_t,  double, double);
REGISTER_TYPE_PROMOTION(uint16_t, double, double);
REGISTER_TYPE_PROMOTION(int32_t,  double, double);
REGISTER_TYPE_PROMOTION(uint32_t, double, double);
REGISTER_TYPE_PROMOTION(int64_t,  double, double);
REGISTER_TYPE_PROMOTION(uint64_t, double, double);

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
REGISTER_BINARY_IMPL(ArithmeticOp::div, /);

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
    return "/";
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
    (ArithmeticOp::add, ArithmeticOp::sub, ArithmeticOp::mul, ArithmeticOp::div, ArithmeticOp::mod),
      (result = arithm_meta<op_static, CPUBackend>::to_string();),
      (result = "InvalidOp";)
  );  // NOLINT(whitespace/parens)
  return result;
}


inline DALIDataType BinaryTypePromotion(DALIDataType left, DALIDataType right) {
  DALIDataType result = DALIDataType::DALI_NO_TYPE;
  TYPE_SWITCH(left, type2id, Left_t,
    (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
        int64_t, uint64_t, float16, float, double),
    (
      TYPE_SWITCH(right, type2id, Right_t,
        (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
            int64_t, uint64_t, float16, float, double),
        (
          using Result_t = binary_result_t<Left_t, Right_t>;
          result = TypeInfo::Create<Result_t>().id();
        ),  // NOLINT(whitespace/parens)
        (DALI_FAIL("Right operand data type not supported, DALIDataType: " +
            std::to_string(right));)
      );   // NOLINT(whitespace/parens)
    ),  // NOLINT(whitespace/parens)
    (DALI_FAIL("Left operand data type not supported, DALIDataType: " + std::to_string(left));)
  );  // NOLINT(whitespace/parens)
  return result;
}

inline DALIDataType TypePromotion(ArithmeticOp op, span<DALIDataType> types) {
  assert(types.size() == 1 || types.size() == 2);
  if (types.size() == 1) {
    return types[0];
  }
  if (op == ArithmeticOp::fdiv) {
    if (!IsFloatingPoint(types[0]) && !IsFloatingPoint(types[1])) {
      return DALIDataType::DALI_FLOAT;
    }
  }
  return BinaryTypePromotion(types[0], types[1]);
}


inline ArithmeticOp NameToOp(const std::string &op_name) {
  static std::map<std::string, ArithmeticOp> token_to_op = {
      {"plus",  ArithmeticOp::plus},
      {"minus", ArithmeticOp::minus},
      {"add",   ArithmeticOp::add},
      {"sub",   ArithmeticOp::sub},
      {"mul",   ArithmeticOp::mul},
      {"div",   ArithmeticOp::div},
      {"fdiv",  ArithmeticOp::fdiv},
      {"mod",   ArithmeticOp::mod}
  };
  auto it = token_to_op.find(op_name);
  DALI_ENFORCE(it != token_to_op.end(), "No implementation for op \"" + op_name + "\".");
  return it->second;
}

inline bool IsScalarLike(const TensorListShape<> &shape) {
  return is_uniform(shape) && shape.sample_dim() == 1 && shape.tensor_shape_span(0)[0] == 1;
}

}  // namespace dali

#endif  // DALI_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_
