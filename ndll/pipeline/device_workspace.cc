#include "ndll/pipeline/device_workspace.h"

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
}

template <>
void DeviceWorkspace::AddInput(shared_ptr<TensorList<GPUBackend>> input) {
  // Save the TensorList
  gpu_inputs_.push_back(input);
  
  // Update the input index map
  input_index_map_.push_back(std::make_pair(false, gpu_inputs_.size()-1));
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
}

template <>
void DeviceWorkspace::AddOutput(shared_ptr<TensorList<GPUBackend>> output) {
  // Save the TensorList
  gpu_outputs_.push_back(output);

  // Update the output index map
  output_index_map_.push_back(std::make_pair(false, gpu_outputs_.size()-1));
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

void DeviceWorkspace::AddParamTensor(shared_ptr<Tensor<CPUBackend>> cpu_tensor,
    shared_ptr<Tensor<GPUBackend>> gpu_tensor) {
  NDLL_ENFORCE(cpu_tensor != nullptr, "Input cpu parameter tensor is nullptr.");
  NDLL_ENFORCE(gpu_tensor != nullptr, "Input gpu parameter tensor is nullptr.");
  cpu_parameters_.push_back(cpu_tensor);
  gpu_parameters_.push_back(gpu_tensor);
}

} // namespace ndll
