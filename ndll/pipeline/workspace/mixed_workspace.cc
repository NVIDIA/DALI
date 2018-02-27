// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/workspace/mixed_workspace.h"

#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

int MixedWorkspace::NumInputAtIdx(int idx) const {
  NDLL_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  if (tensor_meta.first) {
    return cpu_inputs_[tensor_meta.second].size();
  }
  return gpu_inputs_[tensor_meta.second].size();
}

template <>
const Tensor<CPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input with given index does not "
      "have the calling backend type (CPUBackend)");

  NDLL_ENFORCE_VALID_INDEX(data_idx,
      cpu_inputs_[tensor_meta.second].size());

  return *cpu_inputs_[tensor_meta.second][data_idx];
}

template <>
const Tensor<GPUBackend>& MixedWorkspace::Input(int idx, int data_idx) const {
  NDLL_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");

  NDLL_ENFORCE_VALID_INDEX(data_idx,
      gpu_inputs_[tensor_meta.second].size());

  return *gpu_inputs_[tensor_meta.second][data_idx];
}

template <>
TensorList<CPUBackend>* MixedWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
TensorList<GPUBackend>* MixedWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

}  // namespace ndll
