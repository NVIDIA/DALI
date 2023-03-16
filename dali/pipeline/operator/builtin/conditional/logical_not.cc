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
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/conditional/logical_not.h"
#include "dali/pipeline/operator/builtin/conditional/validation.h"

namespace dali {

bool LogicalNot::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);

  EnforceConditionalInputKind(input, name_, "", false);

  output_desc.resize(1);
  output_desc[0] = {input.shape(), DALI_BOOL};

  return true;
}

void LogicalNot::RunImpl(Workspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  TYPE_SWITCH(input.type(), type2id, T, LOGICALLY_EVALUATABLE_TYPES, (
    for (int i = 0; i < output.shape().num_samples(); i++) {
      *output.mutable_tensor<bool>(i) = !(*input.tensor<T>(i));
    }
  ), (DALI_FAIL(make_string("Can't evaluate ", input.type(), " as boolean value to negate it."))));  // NOLINT
}

DALI_SCHEMA(_conditional__Not_)
    .DocStr(R"code(Compute the logical operation ``not``.

This operator is inserted when Python ``not`` statement is used with ``enabled_conditionals=True``.
The inputs are restricted to scalar values of boolean type - this makes them consistent
with Python semantics and allows for unambiguous ``if`` evaluation.
You can use mathematical operator ``==`` and compare with 0 to emulate elementwise equivalent of
logical operation on inputs of other types, shapes, and placements.)code")
    .NumInput(1)
    .NumOutput(1)
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__Not_, LogicalNot, CPU);
DALI_REGISTER_OPERATOR(_conditional__Not_, LogicalNotFailForGpu, GPU);


}  // namespace dali
