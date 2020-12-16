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

void HostWorkspace::GetSample(SampleWorkspace* ws, int data_idx, int thread_idx) {
  DALI_ENFORCE(ws != nullptr, "Input workspace is nullptr.");
  ws->Clear();
  ws->set_data_idx(data_idx);
  ws->set_thread_idx(thread_idx);
  for (const auto& input_meta : input_index_map_) {
    if (input_meta.storage_device == StorageDevice::CPU) {
      ws->AddInput(cpu_inputs_[input_meta.index]->tensor_handle(data_idx));
    } else {
      ws->AddInput(gpu_inputs_[input_meta.index]->tensor_handle(data_idx));
    }
  }
  for (const auto& output_meta : output_index_map_) {
    if (output_meta.storage_device == StorageDevice::CPU) {
      ws->AddOutput(cpu_outputs_[output_meta.index]->tensor_handle(data_idx));
    } else {
      ws->AddOutput(gpu_outputs_[output_meta.index]->tensor_handle(data_idx));
    }
  }
  for (auto& arg_pair : argument_inputs_) {
    assert(!arg_pair.second.should_update);
    ws->AddArgumentInput(arg_pair.first, arg_pair.second.tvec);
  }
}

template <>
const Tensor<CPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  return InputRef<CPUBackend>(idx)[data_idx];
}

template <>
const Tensor<GPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  return InputRef<GPUBackend>(idx)[data_idx];
}

template <>
Tensor<CPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  return OutputRef<CPUBackend>(idx)[data_idx];
}

template <>
Tensor<GPUBackend>& HostWorkspace::Output(int idx, int data_idx) {
  return OutputRef<GPUBackend>(idx)[data_idx];
}

}  // namespace dali
