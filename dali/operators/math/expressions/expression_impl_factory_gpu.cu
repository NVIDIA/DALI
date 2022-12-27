// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_factory_instances/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_impl_gpu.cuh"
#include "dali/operators/math/expressions/expression_tree.h"

namespace dali {
namespace expr {

/**
 * @brief Inspect `expr` to transform runtime information to static information, do the static
 *        type switch to call the OpFactory for given unary op.
 */
std::unique_ptr<ExprImplBase> ExprImplFactoryGpuUnary(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  auto op = NameToOp(expr.GetFuncName());
  VALUE_SWITCH(op, op_static, ALLOWED_UN_OPS, (
          arithm_meta<op_static, GPUBackend> dummy;
          result = OpFactory(dummy, expr);
      ), DALI_FAIL("No suitable op value found"););  // NOLINT(whitespace/parens)
  return result;
}

/**
 * @brief Inspect `expr` to transform runtime information to static information, do the static
 *        type switch to call the OpFactory for given binary op.
 */
std::unique_ptr<ExprImplBase> ExprImplFactoryGpuBinary(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  auto op = NameToOp(expr.GetFuncName());
  VALUE_SWITCH(op, op_static, ALLOWED_BIN_OPS, (
          arithm_meta<op_static, GPUBackend> dummy;
          result = OpFactory(dummy, expr);
      ), DALI_FAIL("No suitable op value found"););  // NOLINT(whitespace/parens)
  return result;
}

/**
 * @brief Inspect `expr` to transform runtime information to static information, do the static
 *        type switch to call the OpFactory for given ternary op.
 */
std::unique_ptr<ExprImplBase> ExprImplFactoryGpuTernary(const ExprFunc &expr) {
  std::unique_ptr<ExprImplBase> result;
  auto op = NameToOp(expr.GetFuncName());
  VALUE_SWITCH(op, op_static, ALLOWED_TERNARY_OPS, (
          arithm_meta<op_static, GPUBackend> dummy;
          result = OpFactory(dummy, expr);
      ), DALI_FAIL("No suitable op value found"););  // NOLINT(whitespace/parens)
  return result;
}

std::unique_ptr<ExprImplBase>
ExprImplFactory(const ExprNode &expr, GPUBackend) {
  DALI_ENFORCE(expr.GetNodeType() == NodeType::Function, "Only function nodes can be executed.");

  switch (expr.GetSubexpressionCount()) {
    case 1:
      return ExprImplFactoryGpuUnary(dynamic_cast<const ExprFunc&>(expr));
    case 2:
      return ExprImplFactoryGpuBinary(dynamic_cast<const ExprFunc&>(expr));
    case 3:
      return ExprImplFactoryGpuTernary(dynamic_cast<const ExprFunc&>(expr));
    default:
      DALI_FAIL("Expressions with " + std::to_string(expr.GetSubexpressionCount()) +
                " subexpressions are not supported. No implementation found.");
  }
}

}  // namespace expr
}  // namespace dali
