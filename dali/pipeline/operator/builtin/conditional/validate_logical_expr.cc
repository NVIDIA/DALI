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

#include <vector>

#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/conditional/validate_logical_expr.h"
#include "dali/pipeline/operator/builtin/conditional/validation.h"

namespace dali {

bool LogicalValidate::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);

  EnforceConditionalInputKind(input, name_, side_, true);

  output_desc.resize(1);
  output_desc[0] = {input.shape(), DALI_BOOL};

  return false;
}

void LogicalValidate::RunImpl(Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  output.ShareData(input);
}

DALI_SCHEMA(_conditional__ValidateLogical)
    .DocStr(
        R"code(Validate the inputs to logical operation ``and`` or ``or`` or the ``if`` condition.

This operator is inserted when Python ``and`` and ``or`` expressions of ``if`` statements are used
with ``enabled_conditionals=True``.
It provides better error messages about the restrictions of inputs to particular expression
rather than falling back to the error reported by the underlying implementation (which is based
on ``if`` statements).
The inputs are restricted to scalar values of boolean type - this makes them consistent
with Python semantics and allows for unambiguous ``if`` evaluation.
You can use mathematical operators `&`, `|`, or ``==`` and compare with 0 to emulate elementwise
equivalent of logical operations on inputs of other types, shapes, and placements.
Note that, in contrast to using logical operators, all subexpressions of elementwise arithmetic
operators are always evaluated.
The GPU variant of this operator is used to fail fast and provide better error message.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("expression_name", "Expression that we are using the validation for: ``and`` or ``or``",
            DALI_STRING)
    .AddArg("expression_side",
            "Which side of the logical expression we are validating: left or right.", DALI_STRING)
    .PassThrough({{{0, 0}}})
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__ValidateLogical, LogicalValidate, CPU);
DALI_REGISTER_OPERATOR(_conditional__ValidateLogical, LogicalFailForGpu, GPU);


}  // namespace dali
