// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/pipeline_debug.h"

namespace dali {
void DLL_PUBLIC PipelineDebug::AddOperator(OpSpec &spec, size_t logical_id) {
  spec.AddArg("max_batch_size", max_batch_size);
  spec.AddArg("device_id", device_id);
  spec.AddArg("num_threads", num_threads);

  std::string device = spec.GetArgument<string>("device");

  if (device == "gpu") {
    gpu_operators.insert({logical_id, DirectOperator<GPUBackend>(spec)});
  } else if (device == "cpu") {
    cpu_operators.insert({logical_id, DirectOperator<CPUBackend>(spec)});
  } else if (device == "mixed") {
    mixed_operators.insert({logical_id, DirectOperator<MixedBackend>(spec)});
  }
}

template <>
std::vector<std::shared_ptr<TensorList<CPUBackend>>> PipelineDebug::RunOperator(
    size_t logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  try {
    return cpu_operators.at(logical_id).Run<CPUBackend, CPUBackend>(inputs, kwargs, &thread_pool);
  } catch (std::out_of_range &e) {
    DALI_FAIL(make_string("Failed to acquire CPU Operator in PipelineDebug. ", e.what()));
  }
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator(
    size_t logical_id, const std::vector<std::shared_ptr<TensorList<GPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  try {
    return gpu_operators.at(logical_id).Run<GPUBackend, GPUBackend>(inputs, kwargs);
  } catch (std::out_of_range &e) {
    DALI_FAIL(make_string("Failed to acquire GPU Operator in PipelineDebug. ", e.what()));
  }
}

template <>
std::vector<std::shared_ptr<TensorList<GPUBackend>>> PipelineDebug::RunOperator(
    size_t logical_id, const std::vector<std::shared_ptr<TensorList<CPUBackend>>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<TensorList<CPUBackend>>> &kwargs) {
  try {
    return mixed_operators.at(logical_id).Run<CPUBackend, GPUBackend>(inputs, kwargs);
  } catch (std::out_of_range &e) {
    DALI_FAIL(make_string("Failed to acquire Mixed Operator in PipelineDebug. ", e.what()));
  }
}
}  // namespace dali