#include "ndll/pipeline/host_workspace.h"

namespace ndll {

template <>
bool HostWorkspace::InputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool HostWorkspace::InputIsType<GPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool HostWorkspace::OutputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool HostWorkspace::OutputIsType<GPUBackend>(int idx) {
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

} // namespace ndll
