// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/device_workspace.h"

#include "ndll/pipeline/sample_workspace.h"

namespace ndll {

template <>
bool DeviceWorkspace::InputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::InputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::OutputIsType<CPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::OutputIsType<GPUBackend>(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return !output_index_map_[idx].first;
}

template <>
const TensorList<CPUBackend>& DeviceWorkspace::Input(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return *cpu_inputs_[tensor_meta.second];
}

template <>
const TensorList<GPUBackend>& DeviceWorkspace::Input(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return *gpu_inputs_[tensor_meta.second];
}

template <>
void DeviceWorkspace::AddInput(shared_ptr<TensorList<CPUBackend>> input) {
  // Save the TensorList
  cpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(true, cpu_inputs_.size()-1));
  cpu_inputs_index_.push_back(input_index_map_.size() - 1);
}

template <>
void DeviceWorkspace::AddInput(shared_ptr<TensorList<GPUBackend>> input) {
  // Save the TensorList
  gpu_inputs_.push_back(input);

  // Update the input index map
  input_index_map_.push_back(std::make_pair(false, gpu_inputs_.size()-1));
  gpu_inputs_index_.push_back(input_index_map_.size() - 1);
}

template <>
void DeviceWorkspace::SetInput(int idx,
    shared_ptr<TensorList<CPUBackend>> input) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());

  // To remove the old input at `idx`, we need to remove it
  // from its typed vector and update the input_index_map
  // entry for all the elements in the vector following it.
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    for (size_t i = tensor_meta.second; i < cpu_inputs_.size(); ++i) {
      int &input_idx = input_index_map_[cpu_inputs_index_[i]].second;
      --input_idx;
    }
    cpu_inputs_.erase(cpu_inputs_.begin() + tensor_meta.second);
    cpu_inputs_index_.erase(cpu_inputs_index_.begin() + tensor_meta.second);
  } else {
    for (size_t i = tensor_meta.second; i < gpu_inputs_.size(); ++i) {
      int &input_idx = input_index_map_[gpu_inputs_index_[i]].second;
      --input_idx;
    }
    gpu_inputs_.erase(gpu_inputs_.begin() + tensor_meta.second);
    gpu_inputs_index_.erase(gpu_inputs_index_.begin() + tensor_meta.second);
  }

  // Now we insert the new input and update its meta data
  cpu_inputs_.push_back(input);
  cpu_inputs_index_.push_back(idx);
  input_index_map_[idx] = std::make_pair(true, cpu_inputs_.size()-1);
}

template <>
void DeviceWorkspace::SetInput(int idx,
    shared_ptr<TensorList<GPUBackend>> input) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());

  // To remove the old input at `idx`, we need to remove it
  // from its typed vector and update the input_index_map
  // entry for all the elements in the vector following it.
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    for (size_t i = tensor_meta.second; i < cpu_inputs_.size(); ++i) {
      int &input_idx = input_index_map_[cpu_inputs_index_[i]].second;
      --input_idx;
    }
    cpu_inputs_.erase(cpu_inputs_.begin() + tensor_meta.second);
    cpu_inputs_index_.erase(cpu_inputs_index_.begin() + tensor_meta.second);
  } else {
    for (size_t i = tensor_meta.second; i < gpu_inputs_.size(); ++i) {
      int &input_idx = input_index_map_[gpu_inputs_index_[i]].second;
      --input_idx;
    }
    gpu_inputs_.erase(gpu_inputs_.begin() + tensor_meta.second);
    gpu_inputs_index_.erase(gpu_inputs_index_.begin() + tensor_meta.second);
  }

  // Now we insert the new input and update its meta data
  gpu_inputs_.push_back(input);
  gpu_inputs_index_.push_back(idx);
  input_index_map_[idx] = std::make_pair(false, gpu_inputs_.size()-1);
}

template <>
TensorList<CPUBackend>* DeviceWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
TensorList<GPUBackend>* DeviceWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

template <>
shared_ptr<TensorList<CPUBackend>> DeviceWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second];
}

template <>
shared_ptr<TensorList<GPUBackend>> DeviceWorkspace::SharedOutput(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second];
}

template <>
void DeviceWorkspace::AddOutput(shared_ptr<TensorList<CPUBackend>> output) {
  // Save the TensorList
  cpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(true, cpu_outputs_.size()-1));
  cpu_outputs_index_.push_back(output_index_map_.size() - 1);
}

template <>
void DeviceWorkspace::AddOutput(shared_ptr<TensorList<GPUBackend>> output) {
  // Save the TensorList
  gpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(false, gpu_outputs_.size()-1));
  gpu_outputs_index_.push_back(output_index_map_.size() - 1);
}

template <>
void DeviceWorkspace::SetOutput(int idx,
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
void DeviceWorkspace::SetOutput(int idx,
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

}  // namespace ndll
