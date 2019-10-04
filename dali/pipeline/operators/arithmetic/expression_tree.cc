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
#include <string>
#include <tuple>
#include <utility>

#include "dali/pipeline/operators/arithmetic/expression_tree.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operators/arithmetic/expression_impl_cpu.h"

namespace dali {

namespace {

// <expr> := <call>|<scalar>|<input>
// <call> := <name> '(' <expr>* ')'
// <name> := fun
// <input> := &uint
// <scalar> := $uint:Type

// add(&0 mul(&1 $0:int8))
// add(&0 rand())

using ParseResult = std::tuple<std::unique_ptr<ExprNode>, int>;

ParseResult ParseExpr(const std::string &expr, int pos);

int ExpectChar(const std::string &expr, int pos, char c) {
  if (expr[pos] == c) {
    return pos + 1;
  }
  DALI_FAIL("Unrecognized token for arithmetic expression: \"" + std::string(1, expr[pos]) +
            "\" at position [" + std::to_string(pos) + "] in: " + expr);
}

int SkipAll(const std::string &expr, int pos, std::function<bool(char c)> predicate) {
  while (predicate(expr[pos])) {
    pos++;
  }
  return pos;
}

int SkipAll(const std::string &expr, int pos, char to_skip) {
  while (expr[pos] == to_skip) {
    pos++;
  }
  return pos;
}

int SkipSpaces(const std::string &expr, int pos) {
  return SkipAll(expr, pos, ' ');
}

int SkipInt(const std::string &expr, int pos) {
  pos = SkipAll(expr, pos, '-');
  return SkipAll(expr, pos, [](char c){ return std::isdigit(c); });
}

std::tuple<int, int> ParseInt(const std::string &expr, int pos) {
  int parsed = atoi(&expr[pos]);
  return std::make_tuple(parsed, SkipInt(expr, pos));
}

std::tuple<std::string, int> ParseName(const std::string &expr, int pos) {
  DALI_ENFORCE(std::isalpha(expr[pos]), "Unrecognized token for arithmetic expression: \"" +
                                            std::string(1, expr[pos]) + "\" at position [" +
                                            std::to_string(pos) + "] in: \"" + expr +
                                            "\". Expected function name to start with character");
  int end_pos = pos + 1;
  while (std::isalnum(expr[end_pos])) {
    end_pos++;
  }
  return std::make_pair(std::string(&expr[pos], &expr[end_pos]), end_pos);
}

ParseResult ParseCall(const std::string &expr, int pos) {
  std::string op;
  std::tie(op, pos) = ParseName(expr, pos);
  auto node = std::make_unique<ExprFunc>(op);
  pos = SkipSpaces(expr, pos);
  pos = ExpectChar(expr, pos, '(');
  pos = SkipSpaces(expr, pos);
  while (expr[pos] != ')') {
    std::unique_ptr<ExprNode> subexpr;
    std::tie(subexpr, pos) = ParseExpr(expr, pos);
    node->AddSubexpression(std::move(subexpr));
    pos = SkipSpaces(expr, pos);
  }
  pos = ExpectChar(expr, pos, ')');
  return std::make_tuple(std::move(node), pos);
}

ParseResult ParseInput(const std::string &expr, int pos) {
  int mapped_input;
  std::tie(mapped_input, pos) = ParseInt(expr, pos);
  auto node = std::make_unique<ExprTensor>(mapped_input);
  return std::make_tuple(std::move(node), pos);
}
ParseResult ParseConstant(const std::string &expr, int pos) {
  // DALI_FAIL("Constant description in expressions is not supported.");
  int mapped_input;
  std::tie(mapped_input, pos) = ParseInt(expr, pos);
  pos = ExpectChar(expr, pos, ':');
  // TODO(klecki): Parse the type encoding
  DALIDataType type = DALIDataType::DALI_INT32;
  while (std::isalnum(expr[pos])) {
    pos++;
  }
  auto node = std::make_unique<ExprConstant>(mapped_input, type);
  return std::make_tuple(std::move(node), pos);
}

ParseResult ParseExpr(const std::string &expr, int pos) {
  if (expr[pos] == '&') {
    return ParseInput(expr, pos + 1);
  } else if (expr[pos] == '$') {
    return ParseConstant(expr, pos + 1);
  } else {
    return ParseCall(expr, pos);
  }
}


}  // namespace

std::unique_ptr<ExprNode> ParseExpressionString(const std::string &expr) {
  return std::move(std::get<0>(ParseExpr(expr, 0)));
}


}  // namespace dali
