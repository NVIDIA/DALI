// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/util.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/builtin/conditional/split.h"
#include "dali/pipeline/operator/builtin/conditional/split_merge.h"
#include "dali/pipeline/operator/builtin/conditional/validation.h"

namespace dali {

template <typename Backend>
bool Split<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.template Input<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  if (if_stmt_implementation_) {
    EnforceConditionalInputKind(predicate, "if", "if-stmt", false);
  }

  DALI_ENFORCE(
      input.num_samples() == predicate.num_samples(),
      make_string("Split description must cover whole input, got ", input.num_samples(),
                  " input samples and ", predicate.num_samples(), " elements denoting the split."));
  DALI_ENFORCE(predicate.shape().sample_dim() == 0, "Only scalar indexing is supported.");

  group_counts_.fill(0);

  for (int i = 0; i < predicate.num_samples(); i++) {
    int output_group_idx = get_group_index(predicate, i);
    group_counts_[output_group_idx]++;
  }

  // TODO(klecki): we can construct the output_desc, it won't be useful now
  return false;
}


template <typename Backend>
void Split<Backend>::RunImpl(Workspace &ws) {
  const auto &input = ws.template Input<Backend>(0);
  const auto &predicate = ws.ArgumentInput("predicate");
  auto sample_idx_in_output = uniform_array<kMaxGroups>(0);

  for (int output_group_idx = 0; output_group_idx < kMaxGroups; output_group_idx++) {
    auto &output = ws.template Output<Backend>(output_group_idx);

    // We can (and need to) do it only once, for each new output instance, when it doesn't have
    // data yet. It should be consistent across iterations.
    if (!output.has_data()) {
      output.set_type(input.type());
      output.set_sample_dim(input.shape().sample_dim());
      output.SetLayout(input.GetLayout());
      output.set_device_id(input.device_id());
      output.set_pinned(input.is_pinned());
      // We let the executor set the desired order, the rest is propagated from the input.
    }
    output.SetSize(group_counts_[output_group_idx]);
  }

  for (int input_sample_idx = 0; input_sample_idx < predicate.num_samples(); input_sample_idx++) {
    int output_group_idx = get_group_index(predicate, input_sample_idx);
    auto &output = ws.template Output<Backend>(output_group_idx);

    // get the output index and increment for the next sample.
    int output_sample_idx = sample_idx_in_output[output_group_idx];
    sample_idx_in_output[output_group_idx]++;

    // share the sample to the output
    output.SetSample(output_sample_idx, input, input_sample_idx);
  }
  WriteTestsDiagnostics(ws);
}


template <typename Backend>
void Split<Backend>::RegisterTestsDiagnostics() {
  this->RegisterDiagnostic("input_pinned", &in_pinned_);
  this->RegisterDiagnostic("output_0_pinned", &out_0_pinned_);
  this->RegisterDiagnostic("output_1_pinned", &out_1_pinned_);
}

template <typename Backend>
void Split<Backend>::WriteTestsDiagnostics(const Workspace &ws) {
  in_pinned_ = ws.template Input<Backend>(0).is_pinned();
  out_0_pinned_ = ws.template Output<Backend>(0).is_pinned();
  out_1_pinned_ = ws.template Output<Backend>(1).is_pinned();
}

DALI_SCHEMA(_conditional__Split)
    .DocStr(R"code(Split batch based on a predicate.)code")
    .NumInput(1)
    .NumOutput(2)
    .AddArg(
        "predicate",
        "Boolean categorization of the input batch. Must be an argument input of scalar values. "
        "Each boolean denotes if the corresponding input sample goes into the true or false "
        "branch.",
        DALI_BOOL, true)
    .AddOptionalArg(
        "_if_stmt",
        "If True, the operator is used as implementation of `if` statement and should apply "
        "additional error checking, presenting the specialized error message. Internal use only.",
        false)
    .SamplewisePassThrough()
    .MakeDocHidden();

DALI_REGISTER_OPERATOR(_conditional__Split, Split<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(_conditional__Split, Split<GPUBackend>, GPU);

}  // namespace dali
