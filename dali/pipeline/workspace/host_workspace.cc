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

#include "dali/pipeline/workspace/host_workspace.h"

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

void HostWorkspace::GetSample(SampleWorkspace *ws,
    int data_idx, int thread_idx) {
  DALI_ENFORCE(ws != nullptr, "Input workspace is nullptr.");
  ws->Clear();
  ws->set_data_idx(data_idx);
  ws->set_thread_idx(thread_idx);
  for (const auto &input_meta : input_index_map_) {
    if (input_meta.first) {
      ws->AddInput(cpu_inputs_[input_meta.second][data_idx]);
    } else {
      ws->AddInput(gpu_inputs_[input_meta.second][data_idx]);
    }
  }
  for (const auto &output_meta : output_index_map_) {
    if (output_meta.first) {
      ws->AddOutput(cpu_outputs_[output_meta.second][data_idx]);
    } else {
      ws->AddOutput(gpu_outputs_[output_meta.second][data_idx]);
    }
  }
  for (auto& arg_pair : argument_inputs_) {
    ws->AddArgumentInput(arg_pair.second, arg_pair.first);
  }
}

int HostWorkspace::NumInputAtIdx(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_inputs_[tensor_meta.second].size();
  }
  return gpu_inputs_[tensor_meta.second].size();
}

int HostWorkspace::NumOutputAtIdx(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_outputs_[tensor_meta.second].size();
  }
  return gpu_outputs_[tensor_meta.second].size();
}

template <>
const Tensor<CPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Input with given index does not "
      "have the calling backend type (CPUBackend)");

  DALI_ENFORCE_VALID_INDEX(data_idx,
      cpu_inputs_[tensor_meta.second].size());

  return *cpu_inputs_[tensor_meta.second][data_idx];
}

template <>
const Tensor<GPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Input TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  DALI_ENFORCE_VALID_INDEX(data_idx,
      gpu_inputs_[tensor_meta.second].size());

  return *gpu_inputs_[tensor_meta.second][data_idx];
}

template <>
Tensor<CPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Output with given index does not "
      "have the calling backend type (CPUBackend)");

  DALI_ENFORCE_VALID_INDEX(data_idx,
      cpu_outputs_[tensor_meta.second].size());

  return *cpu_outputs_[tensor_meta.second][data_idx].get();
}

template <>
Tensor<GPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  DALI_ENFORCE_VALID_INDEX(data_idx,
      gpu_outputs_[tensor_meta.second].size());

  return *gpu_outputs_[tensor_meta.second][data_idx].get();
}

}  // namespace dali
