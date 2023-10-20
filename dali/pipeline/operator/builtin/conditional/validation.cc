// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/operator/builtin/conditional/validation.h"

namespace dali {

namespace {

// TODO(klecki): We should add reductions.all & reductions.any (or similar equivalent) and
// suggest it here.
const char *kSuggestion =
    "\n\nThis input restriction allows the logical expressions to always return scalar boolean "
    "outputs and to be used in unambiguous way in DALI conditionals (`if` statements). "
    "You can set the `dtype` parameter of some operators to `nvidia.dali.types.BOOL` "
    "or use comparison operators which always return booleans.\n"
    "If you need to process inputs of higher dimensionality, different type, or device placement, "
    "you may use bitwise arithmetic operators `&`, `|` or comparison operator (`== 0`) "
    "to emulate logical `and`, `or` and `not` expressions.  Those operations performed on "
    "boolean inputs are equivalent to logical expressions. Keep in mind that, in contrast to using "
    "logical operators, all subexpressions of elementwise arithmetic operators are always evaluated"
    ".";

const char *kAllowedNames[] = {"and", "or", "not", "if"};
const char *kAllowedWheres[] = {"", "left", "right", "if-stmt"};

/**
 * @brief Check if the implementation is using allowed parameters.
 */
void ValidateParamsInternal(const std::string &name, const std::string &where) {
  bool name_is_allowed = false;
  bool where_is_allowed = false;
  for (auto *allowed_name : kAllowedNames) {
    name_is_allowed = name_is_allowed || name == allowed_name;
  }
  for (auto *allowed_where : kAllowedWheres) {
    where_is_allowed = where_is_allowed || where == allowed_where;
  }
  DALI_ENFORCE(name_is_allowed && where_is_allowed,
               "Internal error - DALI diagnostic configured incorrectly.");
}

/**
 * @brief Get the message describing what are the restrictions of input to the if statement
 * or logical expression.
 *
 * @param name One of kAllowedNames, name of the expression of statement that is checked
 * @param enforce_type Whether the input is restricted to booleans.
 */
std::string GetRestrictionMessage(const std::string &name, bool enforce_type) {
  std::string message_start = name == "if" ?
                                  "Conditions inside `if` statements are restricted to" :
                                  make_string("Logical expression `", name, "` is restricted to");

  return make_string(message_start, " scalar (0-d tensors) inputs",
                     (enforce_type ? " of `bool` type" : ""), ", that are placed on CPU.");
}

/**
 * @brief Get the string describing the placement of the input.
 *
 * @param name One of kAllowedNames, name of the expression of statement that is checked
 * @param where One of kAllowedWheres, argument position that is checked
 */
std::string GetSourcePlacementMessage(const std::string &name, const std::string &where) {
  if (name == "if") {
    return " as a condition of the `if` statement.";
  } else if (name == "or" || name == "and") {
    return make_string(" as the ", where, " argument in logical expression.");
  } else {
    return " in logical expression.";
  }
}

}  // namespace

void EnforceConditionalInputKind(const TensorList<CPUBackend> &input, const std::string &name,
                                 const std::string &where, bool enforce_type) {
  ValidateParamsInternal(name, where);

  std::string preamble = GetRestrictionMessage(name, enforce_type);
  std::string where_mention = GetSourcePlacementMessage(name, where);

  auto dim = input.shape().sample_dim();
  DALI_ENFORCE(dim == 0,
               make_string(preamble, " Got a ", dim, "-d input", where_mention, kSuggestion));

  if (enforce_type) {
    auto type = input.type();
    DALI_ENFORCE(type == DALI_BOOL, make_string(preamble, " Got an input of type `", type, "`",
                                                where_mention, kSuggestion));
  }
}

void ReportGpuInputError(const std::string &name, const std::string &where, bool enforce_type) {
  ValidateParamsInternal(name, where);

  std::string preamble = GetRestrictionMessage(name, enforce_type);
  std::string where_mention = GetSourcePlacementMessage(name, where);

  DALI_FAIL(make_string(preamble, " Got a GPU input", where_mention, kSuggestion));
}

}  // namespace dali
