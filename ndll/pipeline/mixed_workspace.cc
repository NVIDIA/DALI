// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/mixed_workspace.h"

#include "ndll/pipeline/sample_workspace.h"

namespace ndll {
namespace internal {

int MixedWorkspace::NumInputAtIdx(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_inputs_[tensor_meta.second].size();
  }
  return gpu_inputs_[tensor_meta.second].size();
}

template <>
bool MixedWorkspace::InputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool MixedWorkspace::InputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool MixedWorkspace::OutputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool MixedWorkspace::OutputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return !output_index_map_[idx].first;
}

template <>
const Tensor<CPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input with given index does not "
      "have the calling backend type (CPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      cpu_inputs_[tensor_meta.second].size());

  return *cpu_inputs_[tensor_meta.second][data_idx];
}

template <>
const Tensor<GPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  NDLL_ENFORCE_VALID_INDEX((size_t)data_idx,
      gpu_inputs_[tensor_meta.second].size());

  return *gpu_inputs_[tensor_meta.second][data_idx];
}

template <>
void MixedWorkspace::AddInput(vector<shared_ptr<Tensor<CPUBackend>>> input) {
  // Save the vector of tensors
  cpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(true, cpu_inputs_.size()-1));
}

template <>
void MixedWorkspace::AddInput(vector<shared_ptr<Tensor<GPUBackend>>> input) {
  // Save the vector of tensors
  gpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(false, gpu_inputs_.size()-1));
}

template <>
TensorList<CPUBackend>* MixedWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
TensorList<GPUBackend>* MixedWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

template <>
shared_ptr<TensorList<CPUBackend>> MixedWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second];
}

template <>
shared_ptr<TensorList<GPUBackend>> MixedWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second];
}

template <>
void MixedWorkspace::AddOutput(shared_ptr<TensorList<CPUBackend>> output) {
  // Save the TensorList
  cpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(true, cpu_outputs_.size()-1));
  cpu_outputs_index_.push_back(output_index_map_.size() - 1);
}

template <>
void MixedWorkspace::AddOutput(shared_ptr<TensorList<GPUBackend>> output) {
  // Save the TensorList
  gpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(false, gpu_outputs_.size()-1));
  gpu_outputs_index_.push_back(output_index_map_.size() - 1);
}

template <>
void MixedWorkspace::SetOutput(int idx,
    shared_ptr<TensorList<CPUBackend>> output) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());

  // To remove the old output at `idx`, we need to remove it
  // from its typed vector and update the output_index_map
  // entry for all the elements in the vector following it.
  auto tensor_meta = output_index_map_[idx];
  if (tensor_meta.first) {
    for (size_t i = tensor_meta.second; i < cpu_outputs_.size(); ++i) {
      int &output_idx = output_index_map_[cpu_outputs_index_[i]].second;
      --output_idx;
    }
    cpu_outputs_.erase(cpu_outputs_.begin() + tensor_meta.second);
    cpu_outputs_index_.erase(cpu_outputs_index_.begin() + tensor_meta.second);
  } else {
    for (size_t i = tensor_meta.second; i < gpu_outputs_.size(); ++i) {
      int &output_idx = output_index_map_[gpu_outputs_index_[i]].second;
      --output_idx;
    }
    gpu_outputs_.erase(gpu_outputs_.begin() + tensor_meta.second);
    gpu_outputs_index_.erase(gpu_outputs_index_.begin() + tensor_meta.second);
  }

  // Now we insert the new output and update its meta data
  cpu_outputs_.push_back(output);
  cpu_outputs_index_.push_back(idx);
  output_index_map_[idx] = std::make_pair(true, cpu_outputs_.size()-1);
}

template <>
void MixedWorkspace::SetOutput(int idx,
    shared_ptr<TensorList<GPUBackend>> output) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());

  // To remove the old output at `idx`, we need to remove it
  // from its typed vector and update the output_index_map
  // entry for all the elements in the vector following it.
  auto tensor_meta = output_index_map_[idx];
  if (tensor_meta.first) {
    for (size_t i = tensor_meta.second; i < cpu_outputs_.size(); ++i) {
      int &output_idx = output_index_map_[cpu_outputs_index_[i]].second;
      --output_idx;
    }
    cpu_outputs_.erase(cpu_outputs_.begin() + tensor_meta.second);
    cpu_outputs_index_.erase(cpu_outputs_index_.begin() + tensor_meta.second);
  } else {
    for (size_t i = tensor_meta.second; i < gpu_outputs_.size(); ++i) {
      int &output_idx = output_index_map_[gpu_outputs_index_[i]].second;
      --output_idx;
    }
    gpu_outputs_.erase(gpu_outputs_.begin() + tensor_meta.second);
    gpu_outputs_index_.erase(gpu_outputs_index_.begin() + tensor_meta.second);
  }

  // Now we insert the new output and update its meta data
  gpu_outputs_.push_back(output);
  gpu_outputs_index_.push_back(idx);
  output_index_map_[idx] = std::make_pair(false, gpu_outputs_.size()-1);
}

}  // namespace internal
}  // namespace ndll
