// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/host_workspace.h"

#include "ndll/pipeline/sample_workspace.h"

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
bool HostWorkspace::InputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool HostWorkspace::InputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool HostWorkspace::OutputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool HostWorkspace::OutputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return !output_index_map_[idx].first;
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
void HostWorkspace::AddInput(vector<shared_ptr<Tensor<CPUBackend>>> input) {
  // Save the vector of tensors
  cpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(true, cpu_inputs_.size()-1));
}

template <>
void HostWorkspace::AddInput(vector<shared_ptr<Tensor<GPUBackend>>> input) {
  // Save the vector of tensors
  gpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(false, gpu_inputs_.size()-1));
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

template <>
vector<shared_ptr<Tensor<CPUBackend>>> HostWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second];
}

template <>
vector<shared_ptr<Tensor<GPUBackend>>> HostWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second];
}

template <>
void HostWorkspace::AddOutput(vector<shared_ptr<Tensor<CPUBackend>>> output) {
  // Save the vector of tensors
  cpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(true, cpu_outputs_.size()-1));
}

template <>
void HostWorkspace::AddOutput(vector<shared_ptr<Tensor<GPUBackend>>> output) {
  // Save the vector of tensors
  gpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(false, gpu_outputs_.size()-1));
}

}  // namespace ndll
