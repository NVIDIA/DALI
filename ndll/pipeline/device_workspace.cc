#include "ndll/pipeline/device_workspace.h"

namespace ndll {

template <>
bool DeviceWorkspace::InputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::InputIsType<GPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::OutputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool DeviceWorkspace::OutputIsType<GPUBackend>(int idx) {
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
Tensor<CPUBackend>* DeviceWorkspace::ParamTensor(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, cpu_parameters_.size());
  return cpu_parameters_[idx].get();
}

template <>
Tensor<GPUBackend>* DeviceWorkspace::ParamTensor(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, gpu_parameters_.size());
  return gpu_parameters_[idx].get();
}

} // namespace ndll
