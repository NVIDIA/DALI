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

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/math/expressions/arithmetic.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_operator_test.h"

namespace dali {

TEST(ExpressionTreeTest, ExpressionTreeNonParsable) {
  ASSERT_THROW(ParseExpressionString("mul"), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("0"), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("mul(0 1)"), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("mul {}"), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("mul "), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("op(op() op()"), std::runtime_error);
  ASSERT_THROW(ParseExpressionString("mul(& &)"), std::runtime_error);
}

TEST(ExpressionTreeTest, ExpressionTree) {
  std::string expr = "mul(&0 &1)";
  auto result = ParseExpressionString(expr);
  auto &result_ref = *result;
  ASSERT_EQ(result_ref.GetNodeType(), NodeType::Function);
  ASSERT_EQ(result_ref.GetSubexpressionCount(), 2);
  ASSERT_EQ(result_ref.GetFuncName(), "mul");
  auto &func_ref = dynamic_cast<ExprFunc&>(result_ref);
  ASSERT_EQ(func_ref[0].GetNodeType(), NodeType::Tensor);
  ASSERT_EQ(func_ref[1].GetNodeType(), NodeType::Tensor);
  ASSERT_EQ(dynamic_cast<ExprTensor &>(func_ref[0]).GetInputIndex(), 0);
  ASSERT_EQ(dynamic_cast<ExprTensor &>(func_ref[1]).GetInputIndex(), 1);
}

TEST(ExpressionTreeTest, ExpressionTreeComplex) {
  std::string expr = "div(sub(&42 &2) &1)";
  auto result = ParseExpressionString(expr);
  auto &result_ref = *result;
  ASSERT_EQ(result_ref.GetNodeType(), NodeType::Function);
  ASSERT_EQ(result_ref.GetSubexpressionCount(), 2);
  EXPECT_EQ(result_ref.GetFuncName(), "div");
  auto &func_ref = dynamic_cast<ExprFunc&>(result_ref);
  ASSERT_EQ(func_ref[0].GetNodeType(), NodeType::Function);
  ASSERT_EQ(func_ref[0].GetSubexpressionCount(), 2);
  EXPECT_EQ(func_ref[0].GetFuncName(), "sub");
  auto &func0_ref = dynamic_cast<ExprFunc&>(func_ref[0]);
  ASSERT_EQ(func0_ref[0].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor &>(func0_ref[0]).GetInputIndex(), 42);
  ASSERT_EQ(func0_ref[1].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor &>(func0_ref[1]).GetInputIndex(), 2);
  ASSERT_EQ(func_ref[1].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor &>(func_ref[1]).GetInputIndex(), 1);
}

}  // namespace dali
