#ifndef NDLL_PIPELINE_DEVICE_WORKSPACE_H_
#define NDLL_PIPELINE_DEVICE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

/**
 * @brief DeviceWorkspace stores all data that a gpu operator operates on, 
 * including its input and output TensorLists, parameter tensors and 
 * meta-data about execution.
 */
class DeviceWorkspace {
public:
  DeviceWorkspace() : stream_(0) {}
  ~DeviceWorkspace() = default;

  /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() const { return input_index_map_.size(); }
  
  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() const { return output_index_map_.size(); }

  /**
   * Returns true if the input TensorList at the given index 
   * has the calling Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx) const;

  /**
   * Returns true if the output TensorList at the given index 
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx) const;
  
  /**
   * @brief Returns the input TensorList at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of 
   * the output at the given index.
   */
  template <typename Backend>
  const TensorList<Backend>& Input(int idx) const;

  /**
   * @brief Adds the input TensorList as an input.
   */
  template <typename Backend>
  void AddInput(shared_ptr<TensorList<Backend>> input);
  
  /**
   * @brief Returns the output TensorList at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of 
   * the output at the given index.
   */
  template <typename Backend>
  TensorList<Backend>* Output(int idx);

  /**
   * @brief Returns the internal shared_ptr to the TensorList at index
   * `idx`.
   *
   * @throws runtime_error If calling type does not match the type of 
   * the output at the given index.
   */
  template <typename Backend>
  shared_ptr<TensorList<Backend>> SharedOutput(int idx);
  
  /**
   * @brief Adds the input TensorList as an output.
   */
  template <typename Backend>
  void AddOutput(shared_ptr<TensorList<Backend>> output);
  
  /**
   * @brief Returns the number of parameter tensors
   */
  inline int NumParamTensor() const { return cpu_parameters_.size(); }
  
  /**
   * @brief Returns the parameter tensor at the given index. The template
   * parameter 'Backend' controls whether the CPU or GPU parameter 
   * tensor is returned.
   */
  template <typename Backend>
  Tensor<Backend>* ParamTensor(int idx);

  /**
   * @brief Adds the input parameter tensors to the workspace.
   */
  void AddParamTensor(shared_ptr<Tensor<CPUBackend>> cpu_tensor,
      shared_ptr<Tensor<GPUBackend>> gpu_tensor);
  
  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const { return stream_; }

private:
  template <typename T>
  using TensorListPtr = shared_ptr<TensorList<T>>;
  vector<TensorListPtr<CPUBackend>> cpu_inputs_, cpu_outputs_;
  vector<TensorListPtr<GPUBackend>> gpu_inputs_, gpu_outputs_;

  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;

  cudaStream_t stream_;
  
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  vector<TensorPtr<CPUBackend>> cpu_parameters_;
  vector<TensorPtr<GPUBackend>> gpu_parameters_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DEVICE_WORKSPACE_H_
