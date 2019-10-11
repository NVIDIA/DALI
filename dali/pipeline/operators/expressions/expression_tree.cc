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

#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/operators/expressions/expression_impl_cpu.h"
#include "dali/pipeline/operators/expressions/expression_tree.h"

namespace dali {

namespace {

using ParseResult = std::tuple<std::unique_ptr<ExprNode>, int>;

ParseResult ParseExpr(const std::string &expr, int pos);

std::string ReportCharacter(char c) {
  if (!std::isprint(c)) {
    return make_string_delim("", "\"<non-printable>\", character code: 0x", std::hex,
                             static_cast<int>(c), std::dec);
  }
  return make_string_delim("", "\"", c, "\", character code: 0x", std::hex, static_cast<int>(c),
                           std::dec);
}

void EnforceNonEnd(const std::string &expr, int pos, const std::string &expected = "") {
  DALI_ENFORCE(pos < static_cast<int>(expr.length()),
               make_string_delim("", "Unexpected end of expression description, expected: ",
                                 expected, " at position [", pos, "] in: ", expr));
}

int ExpectChar(const std::string &expr, int pos, char c) {
  EnforceNonEnd(expr, pos, ReportCharacter(c));
  if (expr[pos] == c) {
    return pos + 1;
  }

  DALI_FAIL(make_string_delim(
      "", "Unrecognized token for expression description: ", ReportCharacter(expr[pos]),
      " at position [", pos, "], expected ", ReportCharacter(c), " in: ", expr));
}

int SkipAll(const std::string &expr, int pos, std::function<bool(char c)> predicate,
            const std::string &expected = "") {
  while (pos < static_cast<int>(expr.length()) && predicate(expr[pos])) {
    pos++;
  }
  return pos;
}

int SkipAll(const std::string &expr, int pos, char to_skip, const std::string &expected = "") {
  while (pos < static_cast<int>(expr.length()) && expr[pos] == to_skip) {
    pos++;
  }
  return pos;
}

int SkipSpaces(const std::string &expr, int pos, const std::string &expected) {
  return SkipAll(expr, pos, ' ', expected);
}

int SkipInt(const std::string &expr, int pos) {
  pos = SkipAll(expr, pos, '-', "numeric value");
  return SkipAll(expr, pos, [](char c) { return std::isdigit(c); }, "numeric value");
}

std::tuple<int, int> ParseInt(const std::string &expr, int pos) {
  EnforceNonEnd(expr, pos, "integer");
  int parsed = atoi(&expr[pos]);
  int new_pos = SkipInt(expr, pos);
  DALI_ENFORCE(pos != new_pos,
               make_string_delim("", "Expected integer value at position [", pos, "] in: ", expr));
  return std::make_tuple(parsed, new_pos);
}

std::tuple<std::string, int> ParseName(const std::string &expr, int pos) {
  EnforceNonEnd(expr, pos, "function name or input description starting with \"&\" or \"$\"");
  DALI_ENFORCE(
      std::isalpha(expr[pos]),
      make_string_delim(
          "", "Unrecognized token for expression description: ", ReportCharacter(expr[pos]),
          " at position [", pos, "] in: \"", expr,
          "\". Expected function name starting with alphabetic character or input description "
          "starting with \"&\" or \"$\"."));
  int end_pos = pos + 1;
  while (end_pos < static_cast<int>(expr.length()) && std::isalnum(expr[end_pos])) {
    end_pos++;
  }
  return std::make_pair(std::string(&expr[pos], &expr[end_pos]), end_pos);
}

ParseResult ParseCall(const std::string &expr, int pos) {
  std::string op;
  std::tie(op, pos) = ParseName(expr, pos);
  auto node = std::make_unique<ExprFunc>(op);
  pos = SkipSpaces(expr, pos, "opening parenthesis \"(\" for subexpression");
  pos = ExpectChar(expr, pos, '(');
  pos = SkipSpaces(expr, pos, "contents of subexpression or closing parenthesis \")\"");
  while (expr[pos] != ')') {
    std::unique_ptr<ExprNode> subexpr;
    std::tie(subexpr, pos) = ParseExpr(expr, pos);
    node->AddSubexpression(std::move(subexpr));
    pos = SkipSpaces(expr, pos, "contents of next subexpression or closing parenthesis \")\"");
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
  EnforceNonEnd(expr, pos, "function name or input description starting with \"&\" or \"$\"");
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
