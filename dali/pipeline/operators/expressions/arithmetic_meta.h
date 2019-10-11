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

#ifndef DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_
#define DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "dali/core/cuda_utils.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

enum class ArithmeticOp : int {
  plus,
  minus,
  add,
  sub,
  mul,
  div,
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
    case ArithmeticOp::mod:
      return 2;
    default:
      return -1;
  }
}

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

inline DALIDataType TypePromotion(DALIDataType left, DALIDataType right) {
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

/**
 * @brief Struct intended as a mapping from ArithmeticOp enum to it's implemetation.
 * It should provide an `impl` static member function of required arity accepting scalar inputs
 * of arbitrary arithmetic types.
 *
 * It also contains input and output counts as well as to_string member function.
 *
 * @tparam op  Mapped Op.
 * @tparam Backend Allows to specialize for given backend
 */
template <ArithmeticOp op, typename Backend>
struct arithm_meta {
  template <typename T>
  DALI_HOST_DEV static constexpr std::enable_if_t<GetOpArity(op) == 1, T> impl(T v);

  template <typename L, typename R>
  DALI_HOST_DEV static constexpr
  std::enable_if_t<GetOpArity(op) == 2, binary_result_t<L, R>> impl(L l, R r);

  static std::string to_string();

  static constexpr int num_inputs = GetOpArity(op);
  static constexpr int num_outputs = 1;
};

#define REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, BACKEND)                       \
  template <>                                                                      \
  template <typename T>                                                            \
  DALI_HOST_DEV constexpr std::enable_if_t<GetOpArity(OP) == 1, T>                 \
  arithm_meta<OP, BACKEND>::impl(T v) {                                            \
    static_assert(GetOpArity(OP) == 1,                                             \
                  "Registered operation arity does not match the requirements.");  \
    return EXPRESSION v;                                                           \
  }                                                                                \
  template <>                                                                      \
  inline std::string arithm_meta<OP, BACKEND>::to_string() {                       \
    return #EXPRESSION;                                                            \
  }

#define REGISTER_UNARY_IMPL(OP, EXPRESSION)               \
  REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, CPUBackend) \
  REGISTER_UNARY_IMPL_BACKEND(OP, EXPRESSION, GPUBackend)

REGISTER_UNARY_IMPL(ArithmeticOp::plus, +);
REGISTER_UNARY_IMPL(ArithmeticOp::minus, -);

#define REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, BACKEND)                          \
  template <>                                                                          \
  template <typename L, typename R>                                                    \
  DALI_HOST_DEV constexpr std::enable_if_t<GetOpArity(OP) == 2, binary_result_t<L, R>> \
  arithm_meta<OP, BACKEND>::impl(L l, R r) {                                           \
    static_assert(GetOpArity(OP) == 2,                                                 \
                  "Registered operation arity does not match the requirements.");      \
    return l EXPRESSION r;                                                             \
  }                                                                                    \
  template <>                                                                          \
  inline std::string arithm_meta<OP, BACKEND>::to_string() {                           \
    return #EXPRESSION;                                                                \
  }

#define REGISTER_BINARY_IMPL(OP, EXPRESSION)               \
  REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, CPUBackend) \
  REGISTER_BINARY_IMPL_BACKEND(OP, EXPRESSION, GPUBackend)

REGISTER_BINARY_IMPL(ArithmeticOp::add, +);
REGISTER_BINARY_IMPL(ArithmeticOp::sub, -);
REGISTER_BINARY_IMPL(ArithmeticOp::mul, *);
REGISTER_BINARY_IMPL(ArithmeticOp::div, /);

template <>
struct arithm_meta<ArithmeticOp::mod, CPUBackend> {
  template <typename L, typename R>
  static constexpr std::enable_if_t<
      std::is_integral<L>::value && std::is_integral<R>::value, binary_result_t<L, R>>
  impl(L l, R r) {
    return l % r;
  }

  template <typename L, typename R>
  static constexpr std::enable_if_t<
      !std::is_integral<L>::value || !std::is_integral<R>::value, binary_result_t<L, R>>
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
  __device__ static constexpr std::enable_if_t<
      std::is_integral<L>::value && std::is_integral<R>::value, binary_result_t<L, R>>
  impl(L l, R r) {
    return l % r;
  }

  template <typename L, typename R>
  __device__ static constexpr std::enable_if_t<
      (!std::is_integral<L>::value || !std::is_integral<R>::value) &&
          (sizeof(L) < sizeof(double) && sizeof(R) < sizeof(double)),
      binary_result_t<L, R>>
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

inline ArithmeticOp NameToOp(const std::string &op_name) {
  static std::map<std::string, ArithmeticOp> token_to_op = {
      std::make_pair("plus",  ArithmeticOp::plus),
      std::make_pair("minus", ArithmeticOp::minus),
      std::make_pair("add",   ArithmeticOp::add),
      std::make_pair("sub",   ArithmeticOp::sub),
      std::make_pair("mul",   ArithmeticOp::mul),
      std::make_pair("div",   ArithmeticOp::div),
      std::make_pair("mod",   ArithmeticOp::mod),
  };
  auto it = token_to_op.find(op_name);
  DALI_ENFORCE(it != token_to_op.end(), "No implementation for op \"" + op_name + "\".");
  return it->second;
}

inline bool IsScalarLike(const kernels::TensorListShape<> &shape) {
  return shape.num_samples() == 1 && shape.num_elements() == 1;
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPRESSIONS_ARITHMETIC_META_H_
