
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_FACTORY_INSTANCES_EXPRESSION_IMPL_FACTORY_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_FACTORY_INSTANCES_EXPRESSION_IMPL_FACTORY_H_

#include <memory>

#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_cpu.h"
#include "dali/core/static_switch.h"

namespace dali {

/**
 * @brief Inspect `expr` to transform runtime information to static information, obtain the
 *        implementation for unary (executor for given expression) and return it.
 *
 * The static type switch goes over input types and input kinds.
 * This is unary case and only tensor inputs are allowed.
 *
 * @tparam ImplTensor template that maps unary Arithmetic Op and input/output type
 *                    to a functor that can execute it over a tile of a tensor (by creating a loop)
 *                    @see ExprImplCpuT or ExprImplGpuT
 * @param expr The expression node for which we return the executor
 */
template <ArithmeticOp op, typename Backend,
          template <ArithmeticOp, typename...> class ImplTensor>
std::unique_ptr<ExprImplBase> ExprImplFactoryUnOp(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  auto input_type = expr[0].GetTypeId();
  TYPE_SWITCH(input_type, type2id, Input_t, ARITHMETIC_ALLOWED_TYPES, (
    using Out_t = typename arithm_meta<op, Backend>::template result_t<Input_t>;
    if (expr[0].GetNodeType() == NodeType::Tensor) {
      result.reset(new ImplTensor<op, Out_t, Input_t>());
    } else {
      DALI_FAIL("Expression cannot have a constant operand");
    }
  ), DALI_FAIL(make_string("Unsupported type: ", input_type)););  // NOLINT(whitespace/parens)
  return result;
}

/**
 * @brief Inspect `expr` to transform runtime information to static information, obtain the
 *        implementation for binary op (executor for given expression) and return it.
 *
 * The static type switch goes over input types and input kinds.
 * This is binary case and three possible input kind combinations are supported:
 * * Tensor and Tensor
 * * Tensor and Constant
 * * Constant and Tensor
 *
 * @tparam ImplTensorTensor template that maps binary Arithmetic Op and input/output types
 *                          to a functor that can execute it over a tile of two tensors.
 *                          The implementation need to loop over inputs and output.
 *                          @see ExprImplCpuTT or ExprImplGpuTT
 * @tparam ImplTensorConstant template that maps binary Arithmetic Op and input/output types
 *                            to a functor that can execute it over a tile of a tensor
 *                            and take a second input as a constant.
 *                            The implementation need to loop over tensor input and output.
 *                            @see ExprImplCpuTC or ExprImplGpuTC
 * @tparam ImplConstantTensor template that maps binary Arithmetic Op and input/output types
 *                            to a functor that can execute it over a tile of a tensor
 *                            but take the first input as a constant.
 *                            The implementation need to loop over tensor input and output.
 *                            @see ExprImplCpuCT or ExprImplGpuCT
 * @param expr The expression node for which we return the executor
 */
template <ArithmeticOp op, typename Backend,
          template <ArithmeticOp, typename...> class ImplTensorTensor,
          template <ArithmeticOp, typename...> class ImplTensorConstant,
          template <ArithmeticOp, typename...> class ImplConstantTensor>
std::unique_ptr<ExprImplBase> ExprImplFactoryBinOp(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  auto left_type = expr[0].GetTypeId();
  auto right_type = expr[1].GetTypeId();
  TYPE_SWITCH(left_type, type2id, Left_t, ARITHMETIC_ALLOWED_TYPES, (
    TYPE_SWITCH(right_type, type2id, Right_t, ARITHMETIC_ALLOWED_TYPES, (
      using Out_t = typename arithm_meta<op, Backend>::template result_t<Left_t, Right_t>;
      if (expr[0].GetNodeType() == NodeType::Tensor && IsScalarLike(expr[1])) {
        result.reset(new ImplTensorConstant<op, Out_t, Left_t, Right_t>());
      } else if (IsScalarLike(expr[0]) && expr[1].GetNodeType() == NodeType::Tensor) {
        result.reset( new ImplConstantTensor<op, Out_t, Left_t, Right_t>());
      } else if (expr[0].GetNodeType() == NodeType::Tensor &&
                 expr[1].GetNodeType() == NodeType::Tensor) {
        // Both are non-scalar tensors
        result.reset(new ImplTensorTensor<op, Out_t, Left_t, Right_t>());
      } else {
        DALI_FAIL("Expression cannot have two scalar operands");
      }
    ), DALI_FAIL(make_string("Invalid type (right operand): ", right_type)););  // NOLINT
  ), DALI_FAIL(make_string("Invalid type (left operarand): ", left_type)););  // NOLINT
  return result;
}


/**
 * @brief Inspect `expr` to transform runtime information to static information, obtain the
 *        implementation for ternary op (executor for given expression) and return it.
 *
 * The static type switch goes over the output type and input kinds.
 * This is ternary case and seven possible input kind combinations are supported,
 * that is all combinations of Tensors and Constants, except (Constant, Constant, Constant)
 *
 * @tparam Impl template that maps binary Arithmetic Op and input/output types
 *                          to a functor that can execute it over a tile of three tensors.
 *                          The implementation need to loop over inputs and output.
 *                          @see ExprImplCpuTernary or ExprImplGpuTernary
 * @param expr The expression node for which we return the executor
 */
template <ArithmeticOp op, typename Backend,
          template <ArithmeticOp, typename, bool, bool, bool> class ImplsAll>
std::unique_ptr<ExprImplBase> ExprImplFactoryTernaryOp(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  int scalar_count = 0;
  bool is_tensor[3];
  for (int i = 0; i < 3; i++) {
    is_tensor[i] = !IsScalarLike(expr[i]);
    if (!is_tensor[i]) {
      scalar_count++;
    }
  }

  // We have a check for this earlier, implementation requires at least one tensor even if it's
  // scalar-like
  if (scalar_count == 3) {
    is_tensor[0] = true;
  }

  TYPE_SWITCH(expr.GetTypeId(), type2id, Out_t, ARITHMETIC_ALLOWED_TYPES, (
    if (is_tensor[0] && is_tensor[1] && is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, true, true, true>());
    } else if (is_tensor[0] && is_tensor[1] && !is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, true, true, false>());
    } else if (is_tensor[0] && !is_tensor[1] && is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, true, false, true>());
    } else if (is_tensor[0] && !is_tensor[1] && !is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, true, false, false>());
    } else if (!is_tensor[0] && is_tensor[1] && is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, false, true, true>());
    } else if (!is_tensor[0] && is_tensor[1] && !is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, false, true, false>());
    } else if (!is_tensor[0] && !is_tensor[1] && is_tensor[2]) {
      result.reset(new ImplsAll<op, Out_t, false, false, true>());
    } else {
      DALI_FAIL("Expression cannot have three scalar operands");
    }
  ), DALI_FAIL("No suitable type found"););  // NOLINT(whitespace/parens)
  return result;
}

#define IMPLEMENT_OP_FACTORY_GPU_UNARY(OP)                                           \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, GPUBackend>, \
                                          const ExprFunc &expr) {                    \
    return ExprImplFactoryUnOp<ArithmeticOp::OP, GPUBackend, ExprImplGpuT>(expr);    \
  }

#define IMPLEMENT_OP_FACTORY_CPU_UNARY(OP)                                           \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, CPUBackend>, \
                                          const ExprFunc &expr) {                    \
    return ExprImplFactoryUnOp<ArithmeticOp::OP, CPUBackend, ExprImplCpuT>(expr);    \
  }

#define IMPLEMENT_OP_FACTORY_GPU_BINARY(OP)                                                 \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, GPUBackend>,        \
                                          const ExprFunc &expr) {                           \
    return ExprImplFactoryBinOp<ArithmeticOp::OP, GPUBackend, ExprImplGpuTT, ExprImplGpuTC, \
                                ExprImplGpuCT>(expr);                                       \
  }

#define IMPLEMENT_OP_FACTORY_CPU_BINARY(OP)                                                 \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, CPUBackend>,        \
                                          const ExprFunc &expr) {                           \
    return ExprImplFactoryBinOp<ArithmeticOp::OP, CPUBackend, ExprImplCpuTT, ExprImplCpuTC, \
                                ExprImplCpuCT>(expr);                                       \
  }

#define IMPLEMENT_OP_FACTORY_CPU_TERNARY(OP)                                                  \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, CPUBackend> op,       \
                                          const ExprFunc &expr) {                             \
    return ExprImplFactoryTernaryOp<ArithmeticOp::OP, CPUBackend, ExprImplCpuTernary>(expr);  \
  }

#define IMPLEMENT_OP_FACTORY_GPU_TERNARY(OP)                                                  \
  std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::OP, GPUBackend> op,       \
                                          const ExprFunc &expr) {                             \
    return ExprImplFactoryTernaryOp<ArithmeticOp::OP, GPUBackend, ExprImplGpuTernary>(expr);  \
  }

/**
 * @brief Factory function returning proper variant of implementation for `plus`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::plus, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `minus`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::minus, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sqrt`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sqrt, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `rsqrt`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::rsqrt, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cbrt`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cbrt, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `exp`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::exp, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log2`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log2, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log10`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log10, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `abs`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::abs, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fabs`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fabs, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `floor`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::floor, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `ceil`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::ceil, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sin`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sin, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cos`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cos, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `tan`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::tan, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `asin`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::asin, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `acos`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::acos, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atan`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atan, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sinh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sinh, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cosh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cosh, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `tanh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::tanh, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `asinh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::asinh, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `acosh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::acosh, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atanh`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atanh, CPUBackend>,
                                        const ExprFunc &expr);



/**
 * @brief Factory function returning proper variant of implementation for `add`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::add, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sub`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sub, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `mul`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::mul, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `div`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::div, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fdiv`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fdiv, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `mod`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::mod, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `min`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::min, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `max`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::max, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `pow`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::pow, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fpow`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fpow, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atan2`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atan2, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `eq`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::eq, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `neq`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::neq, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `lt`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::lt, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `leq`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::leq, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `gt`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::gt, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `geq`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::geq, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_and`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_and, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_or`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_or, CPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_xor`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_xor, CPUBackend>,
                                        const ExprFunc &expr);


/**
 * @brief Factory function returning proper variant of implementation for `clamp`
 *        on CPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::clamp, CPUBackend>,
                                        const ExprFunc &expr);



/**
 * @brief Factory function returning proper variant of implementation for `plus`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::plus, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `minus`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::minus, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sqrt`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sqrt, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `rsqrt`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::rsqrt, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cbrt`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cbrt, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `exp`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::exp, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log2`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log2, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `log10`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::log10, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `abs`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::abs, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fabs`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fabs, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `floor`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::floor, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `ceil`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::ceil, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sin`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sin, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cos`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cos, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `tan`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::tan, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `asin`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::asin, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `acos`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::acos, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atan`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atan, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sinh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sinh, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `cosh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::cosh, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `tanh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::tanh, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `asinh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::asinh, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `acosh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::acosh, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atanh`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atanh, GPUBackend>,
                                        const ExprFunc &expr);



/**
 * @brief Factory function returning proper variant of implementation for `add`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::add, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `sub`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::sub, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `mul`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::mul, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `div`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::div, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fdiv`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fdiv, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `mod`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::mod, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `min`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::min, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `max`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::max, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `pow`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::pow, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `fpow`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::fpow, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `atan2`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::atan2, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `eq`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::eq, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `neq`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::neq, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `lt`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::lt, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `leq`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::leq, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `gt`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::gt, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `geq`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::geq, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_and`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_and, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_or`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_or, GPUBackend>,
                                        const ExprFunc &expr);

/**
 * @brief Factory function returning proper variant of implementation for `bit_xor`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::bit_xor, GPUBackend>,
                                        const ExprFunc &expr);
/**
 * @brief Factory function returning proper variant of implementation for `clamp`
 *        on GPUBackend for supplied input types and input kinds (Scalar/Tensor inputs),
 *        specified in `expr`.
 */
std::unique_ptr<ExprImplBase> OpFactory(arithm_meta<ArithmeticOp::clamp, GPUBackend>,
                                        const ExprFunc &expr);

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_FACTORY_INSTANCES_EXPRESSION_IMPL_FACTORY_H_
