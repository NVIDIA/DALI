#include "ndll/pipeline/workspace.h"

namespace ndll {

// Note: We could just macros these instead of writing it all out...

const Tensor<CPUBackend>& Workspace::Input(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < cpu_inputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(cpu_inputs_.size()) + ")");
  // Wrap the Tensor at data_idx_ in the specified input tensor
  cpu_input_wrappers_[idx][data_idx_].ShareData(cpu_inputs_[idx], data_idx_);
  return cpu_input_wrappers_[idx][data_idx_];
}

Tensor<CPUBackend>* Workspace::Output(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < cpu_outputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(cpu_outputs_.size()) + ")");
  // Wrap the Tensor at data_idx_ in the specified output tensor
  cpu_output_wrappers_[idx][data_idx_].ShareData(cpu_outputs_[idx], data_idx_);
  return &cpu_output_wrappers_[idx][data_idx_];
}

const Tensor<GPUBackend>& Workspace::GPUInput(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < gpu_inputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(gpu_inputs_.size()) + ")");
  // Wrap the Tensor at data_idx_ in the specified input tensor
  gpu_input_wrappers_[idx][data_idx_].ShareData(gpu_inputs_[idx], data_idx_);
  return gpu_input_wrappers_[idx][data_idx_];
}

Tensor<GPUBackend>* Workspace::GPUOutput(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < gpu_outputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(gpu_outputs_.size()) + ")");
  // Wrap the Tensor at data_idx_ in the specified output tensor
  gpu_output_wrappers_[idx][data_idx_].ShareData(gpu_outputs_[idx], data_idx_);
  return &gpu_output_wrappers_[idx][data_idx_];
}

const TensorList<CPUBackend>& Workspace::InputList(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < cpu_inputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(cpu_inputs_.size()) + ")");
  return *cpu_inputs_[idx];
}

TensorList<CPUBackend>* Workspace::OutputList(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < cpu_outputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(cpu_outputs_.size()) + ")");
  return cpu_outputs_[idx];  
}

const TensorList<GPUBackend>& Workspace::GPUInputList(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < gpu_inputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(gpu_inputs_.size()) + ")");
  return *gpu_inputs_[idx];
}

TensorList<GPUBackend>* Workspace::GPUOutputList(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < gpu_outputs_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(gpu_outputs_.size()) + ")");
  return gpu_outputs_[idx];
}

Tensor<CPUBackend>* Workspace::ParamTensor(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < cpu_parameters_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(cpu_parameters_.size()) + ")");
  return &cpu_parameters_[idx];
}

Tensor<GPUBackend>* Workspace::GPUParamTensor(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < gpu_parameters_.size(), "Index out of range." +
      std::to_string(idx) + " not in range [0, " +
      std::to_string(gpu_parameters_.size()) + ")");
  return &gpu_parameters_[idx];
}

} // namespace ndll
