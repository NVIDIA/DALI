// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

template <>
const Tensor<CPUBackend>& SampleWorkspace::Input(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_index_map_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_index_map_.size())
      + ")");

  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input Tensor with given "
      "index does not have the calling backend type (CPUBackend)");
  return *cpu_inputs_[tensor_meta.second];
}

template <>
const Tensor<GPUBackend>& SampleWorkspace::Input(int idx) const {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < input_index_map_.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(input_index_map_.size())
      + ")");

  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (GPUBackend)");
  return *gpu_inputs_[tensor_meta.second];
}

template <>
Tensor<CPUBackend>* SampleWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX(idx, output_index_map_.size());

  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
Tensor<GPUBackend>* SampleWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX(idx, output_index_map_.size());

  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output Tensor with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

}  // namespace ndll
