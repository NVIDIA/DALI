// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/workspace/host_workspace.h"

#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

void HostWorkspace::GetSample(SampleWorkspace *ws,
    int data_idx, int thread_idx) {
  NDLL_ENFORCE(ws != nullptr, "Input workspace is nullptr.");
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
}

int HostWorkspace::NumInputAtIdx(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_inputs_[tensor_meta.second].size();
  }
  return gpu_inputs_[tensor_meta.second].size();
}

int HostWorkspace::NumOutputAtIdx(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_outputs_[tensor_meta.second].size();
  }
  return gpu_outputs_[tensor_meta.second].size();
}

template <>
const Tensor<CPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input with given index does not "
      "have the calling backend type (CPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      cpu_inputs_[tensor_meta.second].size());

  return *cpu_inputs_[tensor_meta.second][data_idx];
}

template <>
const Tensor<GPUBackend>& HostWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Input TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      gpu_inputs_[tensor_meta.second].size());

  return *gpu_inputs_[tensor_meta.second][data_idx];
}

template <>
Tensor<CPUBackend>* HostWorkspace::Output(int idx, int data_idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output with given index does not "
      "have the calling backend type (CPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      cpu_outputs_[tensor_meta.second].size());

  return cpu_outputs_[tensor_meta.second][data_idx].get();
}

template <>
Tensor<GPUBackend>* HostWorkspace::Output(int idx, int data_idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      gpu_outputs_[tensor_meta.second].size());

  return gpu_outputs_[tensor_meta.second][data_idx].get();
}

}  // namespace ndll
