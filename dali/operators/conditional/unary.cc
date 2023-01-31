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
#include "dali/operators/conditional/unary.h"
#include "dali/operators/conditional/validation.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

bool LogicalNot::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);

  // TODO(klecki): Lift this restriction for input type, as it should be safe to do.
  // Do it at the same time as lifting such restriction for the Split operator `predicate`
  // that implements `if` statement, so we do not introduce the `if not not x` idiom.
  EnforceConditionalInputKind(input, name_, "", true);

  output_desc.resize(1);
  output_desc[0] = {input.shape(), DALI_BOOL};

  return true;
}

void LogicalNot::RunImpl(Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  for (int i = 0; i < output.shape().num_samples(); i++) {
    *output.mutable_tensor<bool>(i) = !(*input.tensor<bool>(i));
  }
}

DALI_SCHEMA(_conditional__Not)
    .DocStr(R"code(Compute the logical operation ``not``.

This operator is inserted when Python ``and`` statement is used with ``enabled_conditionals=True``.
The inputs are restricted to scalar values of boolean type - this makes them consistent
with Python semantics and allows for unambiguous ``if`` evaluation.
You can use mathematical operator ``&`` to emulate elementwise equivalent of logical operation on
inputs of other types and shapes.)code")
    .NumInput(1)
    .NumOutput(1)
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__And, LogicalNot, CPU);


}  // namespace dali
