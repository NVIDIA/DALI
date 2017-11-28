#include "ndll/pipeline/batch_workspace.h"

namespace ndll {

template <>
bool BatchWorkspace::InputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return input_index_map_[idx].first;
}

template <>
bool BatchWorkspace::InputIsType<GPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  return !input_index_map_[idx].first;
}

template <>
bool BatchWorkspace::OutputIsType<CPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return output_index_map_[idx].first;
}

template <>
bool BatchWorkspace::OutputIsType<GPUBackend>(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  return !output_index_map_[idx].first;
}

template <>
const TensorList<CPUBackend>& BatchWorkspace::Input(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Input TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return *cpu_inputs_[tensor_meta.second];
}

template <>
const TensorList<GPUBackend>& BatchWorkspace::Input(int idx) const {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, input_index_map_.size());
  auto tensor_meta = input_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return *gpu_inputs_[tensor_meta.second];
}

template <>
TensorList<CPUBackend>* BatchWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (CPUBackend)");
  return cpu_outputs_[tensor_meta.second].get();
}

template <>
TensorList<GPUBackend>* BatchWorkspace::Output(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
  auto tensor_meta = output_index_map_[idx];
  NDLL_ENFORCE(!tensor_meta.first, "Output TensorList with given "
      "index does not have the calling backend type (GPUBackend)");
  return gpu_outputs_[tensor_meta.second].get();
}

template <>
Tensor<CPUBackend>* BatchWorkspace::ParamTensor(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, cpu_parameters_.size());
  return cpu_parameters_[idx].get();
}

template <>
Tensor<GPUBackend>* BatchWorkspace::ParamTensor(int idx) {
  NDLL_ENFORCE_VALID_INDEX((size_t)idx, gpu_parameters_.size());
  return gpu_parameters_[idx].get();
}

} // namespace ndll
