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

#include <memory>

#include "dali/pipeline/operators/arithmetic/expression_tree.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operators/arithmetic/expression_impl_gpu.cuh"
#include "dali/pipeline/operators/arithmetic/expression_impl_factory.h"

// float16
#define ALLOWED_TYPES \
  (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double)

#define ALLOWED_UN_OPS \
  (ArithmeticOp::plus, ArithmeticOp::minus)

#define ALLOWED_BIN_OPS \
  (ArithmeticOp::add, ArithmeticOp::sub, ArithmeticOp::mul, ArithmeticOp::div, ArithmeticOp::mod)

namespace dali {

std::unique_ptr<ExpressionImplBase> ExprImplFactory(const DeviceWorkspace &ws,
                                                    const ExprNode &expr) {
  std::unique_ptr<ExpressionImplBase> result;
  DALI_ENFORCE(expr.GetNodeType() == NodeType::Function, "Only function nodes can be executed.");

  switch (expr.GetSubexpressionCount()) {
    case 1: {
      auto op = NameToOp(expr.GetOp());
      auto input_type = expr[0].GetTypeId();
      TYPE_SWITCH(input_type, type2id, Input_t, ALLOWED_TYPES, (
            VALUE_SWITCH(op, op_static, ALLOWED_UN_OPS, (
              using Out_t = Input_t;
              if (expr[0].GetNodeType() == NodeType::Tensor) {
                using ExprImpl = ExpressionImplBinGPU<op_static,
                    GPUUnTileDesc<Out_t, InputDescriptor<Input_t, true>>, GPUUnTileStorage>;
                result.reset(new ExprImpl());
              } else {
                DALI_FAIL("Unary expression can have only tensor input, "
                          "constant input encountered.");
              }
          ), DALI_FAIL("No suitable op value found"););  // NOLINT(whitespace/parens)
      ), DALI_FAIL("No suitable type found"););  // NOLINT(whitespace/parens)
      return result;
    }

    case 2: {
      auto op = NameToOp(expr.GetOp());
      auto left_type = expr[0].GetTypeId();
      auto right_type = expr[1].GetTypeId();

      // 4-fold static switch
      TYPE_SWITCH(left_type, type2id, Left_t, ALLOWED_TYPES, (
        TYPE_SWITCH(right_type, type2id, Right_t, ALLOWED_TYPES, (
            VALUE_SWITCH(op, op_static, ALLOWED_BIN_OPS, (
              using Out_t = binary_result_t<Left_t, Right_t>;
              if (expr[0].GetNodeType() == NodeType::Tensor &&
                  expr[1].GetNodeType() == NodeType::Tensor) {
                result.reset(new ExpressionImplBinGPU<op_static, GPUBinTileDesc<Out_t,
                    InputDescriptor<Left_t, true>,
                    InputDescriptor<Right_t, true>>, GPUBinTileStorage>());
              } else if (expr[0].GetNodeType() == NodeType::Tensor &&
                         expr[1].GetNodeType() != NodeType::Tensor) {
                result.reset(new ExpressionImplBinGPU<op_static, GPUBinTileDesc<Out_t,
                    InputDescriptor<Left_t, true>,
                    InputDescriptor<Right_t, false>>, GPUBinTileStorage>());

              } else if (expr[0].GetNodeType() != NodeType::Tensor &&
                         expr[1].GetNodeType() == NodeType::Tensor) {
                result.reset(new ExpressionImplBinGPU<op_static, GPUBinTileDesc<Out_t,
                    InputDescriptor<Left_t, false>,
                    InputDescriptor<Right_t, true>>, GPUBinTileStorage>());
              } else {
                DALI_FAIL("Expression cannot have two constant operands");
              }
          ), DALI_FAIL("No suitable op value found"););  // NOLINT(whitespace/parens)
        ), DALI_FAIL("No suitable type found"););  // NOLINT(whitespace/parens)
      ), DALI_FAIL("No suitable type found"););  // NOLINT(whitespace/parens)
      return result;
    }
    default:
      DALI_FAIL("Expressions with " + std::to_string(expr.GetSubexpressionCount()) +
                " subexpressions are not supported. No implemetation found.");
      break;
  }
}

}  // namespace dali
