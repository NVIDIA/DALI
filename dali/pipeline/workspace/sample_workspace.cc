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

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

template <>
const Tensor<CPUBackend>& SampleWorkspace::Input(int idx) const {
  DALI_ENFORCE(idx >= 0, "Negative index not supported.");
  DALI_ENFORCE((size_t)idx < input_index_map_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_index_map_.size())
      + ")");

  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Input Tensor with given "
      "index does not have the calling backend type (CPUBackend)");
  return *cpu_inputs_[tensor_meta.second];
}

template <>
const Tensor<GPUBackend>& SampleWorkspace::Input(int idx) const {
  DALI_ENFORCE(idx >= 0, "Negative index not supported.");
  DALI_ENFORCE((size_t)idx < input_index_map_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_index_map_.size())
      + ")");

  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (GPUBackend)");
  return *gpu_inputs_[tensor_meta.second];
}

template <>
Tensor<CPUBackend>* SampleWorkspace::Output(int idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());

  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (CPUBackend)");
  cpu_outputs_[tensor_meta.second]->reset_reference_count();
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
Tensor<GPUBackend>* SampleWorkspace::Output(int idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());

  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (GPUBackend)");
  gpu_outputs_[tensor_meta.second]->reset_reference_count();
  return gpu_outputs_[tensor_meta.second].get();
}

}  // namespace dali
