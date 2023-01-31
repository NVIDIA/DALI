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
#include "dali/operators/conditional/binary.h"
#include "dali/operators/conditional/validation.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"

namespace dali {

bool BinaryLogicalOp::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &left = ws.template Input<CPUBackend>(0);
  const auto &right = ws.template Input<CPUBackend>(1);

  EnforceConditionalInputKind(left, name_, "left");
  EnforceConditionalInputKind(right, name_, "right");

  output_desc.resize(1);
  output_desc[0] = {left.shape(), left.type()};

  return true;
}

void BinaryLogicalOp::RunImpl(Workspace &ws) {
  const auto &left = ws.template Input<CPUBackend>(0);
  const auto &right = ws.template Input<CPUBackend>(1);
  auto &output = ws.template Output<CPUBackend>(0);
  for (int i = 0; i < output.shape().num_samples(); i++) {
    *output.mutable_tensor<bool>(i) = compute(*left.tensor<bool>(i), *right.tensor<bool>(i));
  }
}

DALI_SCHEMA(_conditional__And)
    .DocStr(R"code(Compute the logical operation ``and`` between the two inputs.

This operator is inserted when Python ``and`` statement is used with ``enabled_conditionals=True``.
The inputs are restricted to scalar values of boolean type - this makes them consistent
with Python semantics and allows for unambiguous ``if`` evaluation.
You can use arithmetic operator ``&`` to emulate elementwise logical operations on inputs of other
types and shapes. Bitwise operator used with boolean inputs is equivalent to logical operation.
)code")
    .NumInput(2)
    .NumOutput(1)
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__And, LogicalAnd, CPU);

DALI_SCHEMA(_conditional__Or)
    .DocStr(R"code(Compute the logical operation ``or`` between the two inputs.

This operator is inserted when Python ``or`` statement is used with ``enabled_conditionals=True``.
The inputs are restricted to scalar values of boolean type - this makes them consistent
with Python semantics and allows for unambiguous ``if`` evaluation.
You can use arithmetic operator ``|`` to emulate elementwise logical operations on inputs of other
types and shapes. Bitwise operator used with boolean inputs is equivalent to logical operation.
)code")
    .NumInput(2)
    .NumOutput(1)
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__Or, LogicalOr, CPU);

}  // namespace dali
