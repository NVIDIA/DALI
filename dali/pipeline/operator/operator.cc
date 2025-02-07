// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/conditional/split_merge.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

inline bool IsVideoInput(const OpSchema& schema) {
#if FFMPEG_ENABLED
  return schema.name() == "experimental__inputs__Video";
#else
  return false;
#endif
}

void OperatorBase::EnforceUniformInputBatchSize(const Workspace &ws) const {
  // Builtin operators have relaxed checks for the purpose of conditional execution
  if (IsSplitOrMerge(spec_.GetSchema())) {
    return;
  }
  if (IsVideoInput(spec_.GetSchema())) {
    return;
  }
  auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumInput(); i++) {
    DALI_ENFORCE(curr_batch_size == ws.GetInputBatchSize(i),
                 "Batch size has to be uniform across one iteration.");
  }
  for (const auto &arg : ws.ArgumentInputs()) {
    DALI_ENFORCE(
        curr_batch_size == static_cast<decltype(curr_batch_size)>(arg.cpu->num_samples()),
        "ArgumentInput has to have the same batch size as an input.");
  }
}

void OperatorBase::EnforceUniformOutputBatchSize(const Workspace &ws) const {
  // Builtin operators have relaxed checks for the purpose of conditional execution
  if (IsSplitOrMerge(spec_.GetSchema())) {
    return;
  }
  if (IsVideoInput(spec_.GetSchema())) {
    return;
  }
  auto ref_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumOutput(); i++) {
    auto output_batch_size = ws.GetOutputBatchSize(i);
    DALI_ENFORCE(ref_batch_size == output_batch_size,
                 make_string("Batch size has to be uniform across one iteration. Expected: ",
                             ref_batch_size, "; Actual: ", output_batch_size));
  }
}

DALI_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);

std::unique_ptr<OperatorBase> InstantiateOperator(const OpSpec &spec) {
  string device = spec.GetArgument<string>("device");
  // traverse devices by likelihood (gpu, cpu, mixed, support)
  if (device == "gpu") {
    return GPUOperatorRegistry::Registry().Create(spec.SchemaName(), spec, &device);
  } else if (device == "cpu") {
    return CPUOperatorRegistry::Registry().Create(spec.SchemaName(), spec, &device);
  } else if (device == "mixed") {
    return MixedOperatorRegistry::Registry().Create(spec.SchemaName(), spec, &device);
  } else {
    DALI_FAIL("Unknown device: " + device);
  }
}

}  // namespace dali
