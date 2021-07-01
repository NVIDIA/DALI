// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
void OperatorBase::EnforceUniformInputBatchSize(const workspace_t<Backend> &ws) const {
  auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumInput(); i++) {
    DALI_ENFORCE(curr_batch_size == ws.GetInputBatchSize(i),
                 "Batch size has to be uniform across one iteration.");
  }
  const ArgumentWorkspace &argument_ws = ws;
  for (const auto &arg : argument_ws) {
    DALI_ENFORCE(
        curr_batch_size == static_cast<decltype(curr_batch_size)>(arg.second.tvec->ntensor()),
        "ArgumentInput has to have the same batch size as an input.");
  }
}


template <typename Backend>
void OperatorBase::EnforceUniformOutputBatchSize(const workspace_t<Backend> &ws) const {
  auto ref_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumOutput(); i++) {
    auto output_batch_size = ws.template OutputRef<Backend>(i).shape().num_samples();
    DALI_ENFORCE(ref_batch_size == output_batch_size,
                 make_string("Batch size has to be uniform across one iteration. Expected: ",
                             ref_batch_size, "; Actual: ", output_batch_size));
  }
}


template <>
void OperatorBase::EnforceUniformOutputBatchSize<MixedBackend>(
    const workspace_t<MixedBackend> &ws) const {
  auto ref_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : ws.GetRequestedBatchSize(0);
  for (int i = 0; i < ws.NumOutput(); i++) {
    auto output_batch_size = const_cast<workspace_t<MixedBackend> &>(ws)
                                 .template Output<GPUBackend>(i)
                                 .shape()
                                 .num_samples();
    DALI_ENFORCE(ref_batch_size == output_batch_size,
                 make_string("Batch size has to be uniform across one iteration. Expected: ",
                             ref_batch_size, "; Actual: ", output_batch_size));
  }
}


template void OperatorBase::EnforceUniformInputBatchSize<CPUBackend>(const workspace_t<CPUBackend> &w) const;  // NOLINT
template void OperatorBase::EnforceUniformInputBatchSize<GPUBackend>(const workspace_t<GPUBackend> &w) const;  // NOLINT
template void OperatorBase::EnforceUniformInputBatchSize<MixedBackend>(const workspace_t<MixedBackend> &w) const;  // NOLINT
template void OperatorBase::EnforceUniformOutputBatchSize<CPUBackend>(const workspace_t<CPUBackend> &w) const;  // NOLINT
template void OperatorBase::EnforceUniformOutputBatchSize<GPUBackend>(const workspace_t<GPUBackend> &w) const;  // NOLINT


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
