// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/workspace/device_workspace.h"

#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

template <>
const TensorList<CPUBackend>& DeviceWorkspace::Input(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Input TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return *cpu_inputs_[tensor_meta.second];
}

template <>
const TensorList<GPUBackend>& DeviceWorkspace::Input(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return *gpu_inputs_[tensor_meta.second];
}

template <>
TensorList<CPUBackend>* DeviceWorkspace::Output(int idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
TensorList<GPUBackend>* DeviceWorkspace::Output(int idx) {
  DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  DALI_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

}  // namespace dali
