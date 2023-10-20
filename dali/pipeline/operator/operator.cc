// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if FFMPEG_ENABLED
#include "dali/operators/input/video_input.h"
#endif
#include "dali/pipeline/operator/builtin/conditional/split_merge.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
void OperatorBase::EnforceUniformInputBatchSize(const Workspace &ws) const {
  // Builtin operators have relaxed checks for the purpose of conditional execution
  if (IsSplitOrMerge(spec_.GetSchema())) {
    return;
  }
#if FFMPEG_ENABLED
  // VideoInput operator has relaxed check, since it actually creates a batch, therefore
  // the batch size might change. It is a perfectly fine behaviour.
  if (IsVideoInput(spec_.GetSchema())) {
    return;
  }
#endif
  auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumInput(); i++) {
    DALI_ENFORCE(curr_batch_size == ws.GetInputBatchSize(i),
                 "Batch size has to be uniform across one iteration.");
  }
  const ArgumentWorkspace &argument_ws = ws;
  for (const auto &arg : argument_ws) {
    DALI_ENFORCE(
        curr_batch_size == static_cast<decltype(curr_batch_size)>(arg.second.tvec->num_samples()),
        "ArgumentInput has to have the same batch size as an input.");
  }
}


template <typename Backend>
void OperatorBase::EnforceUniformOutputBatchSize(const Workspace &ws) const {
  // Builtin operators have relaxed checks for the purpose of conditional execution
  if (IsSplitOrMerge(spec_.GetSchema())) {
    return;
  }
#if FFMPEG_ENABLED
  // VideoInput operator has relaxed check, since it actually creates a batch, therefore
  // the batch size might change. It is a perfectly fine behaviour.
  if (IsVideoInput(spec_.GetSchema())) {
    return;
  }
#endif
  auto ref_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumOutput(); i++) {
    auto output_batch_size = ws.Output<Backend>(i).shape().num_samples();
    DALI_ENFORCE(ref_batch_size == output_batch_size,
                 make_string("Batch size has to be uniform across one iteration. Expected: ",
                             ref_batch_size, "; Actual: ", output_batch_size));
  }
}


template <>
void OperatorBase::EnforceUniformOutputBatchSize<MixedBackend>(
    const Workspace &ws) const {
  auto ref_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumOutput(); i++) {
    auto output_batch_size = const_cast<Workspace &>(ws)
                                 .Output<GPUBackend>(i)
                                 .shape()
                                 .num_samples();
    DALI_ENFORCE(ref_batch_size == output_batch_size,
                 make_string("Batch size has to be uniform across one iteration. Expected: ",
                             ref_batch_size, "; Actual: ", output_batch_size));
  }
}


template void OperatorBase::EnforceUniformInputBatchSize<CPUBackend>(const Workspace &w) const;  // NOLINT
template void OperatorBase::EnforceUniformInputBatchSize<GPUBackend>(const Workspace &w) const;  // NOLINT
template void OperatorBase::EnforceUniformInputBatchSize<MixedBackend>(const Workspace &w) const;  // NOLINT
template void OperatorBase::EnforceUniformOutputBatchSize<CPUBackend>(const Workspace &w) const;  // NOLINT
template void OperatorBase::EnforceUniformOutputBatchSize<GPUBackend>(const Workspace &w) const;  // NOLINT


DALI_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);

std::unique_ptr<OperatorBase> InstantiateOperator(const OpSpec &spec) {
  string device = spec.GetArgument<string>("device");
  // traverse devices by likelihood (gpu, cpu, mixed, support)
  if (device == "gpu") {
    return GPUOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else if (device == "cpu") {
    return CPUOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else if (device == "mixed") {
    return MixedOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else {
    DALI_FAIL("Unknown device: " + device);
  }
}

}  // namespace dali
