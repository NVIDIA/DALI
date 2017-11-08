#ifndef NDLL_PIPELINE_WORKSPACE_H_
#define NDLL_PIPELINE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

class Workspace {
public:
  Workspace() {}
  ~Workspace() = default;

  /**
   * @brief Returns Tensor with index = data_idx() from the input
   * TensorList at index = `idx`.
   */
  const Tensor<CPUBackend>& Input(int idx);

  /**
   * @brief Returns Tensor with index = data_idx() from the output
   * TensorList at index = `idx`.
   */
  Tensor<CPUBackend>* Output(int idx);

  /**
   * @brief Returns Tensor with index = `data_idx()` from the input
   * GPU TensorList at index = `idx`.
   */
  const Tensor<GPUBackend>& GPUInput(int idx);

  /**
   * @brief Returns Tensor with index = data_idx() from the output
   * GPU TensorList at index = `idx`.
   */
  Tensor<GPUBackend>* GPUOutput(int idx);

  /**
   * @brief Returns the input TensorList with index = idx. 
   */
  const TensorList<CPUBackend>& InputList(int idx);

  /**
   * @brief Returns the output TensorList with index = idx. 
   */
  TensorList<CPUBackend>* OutputList(int idx);

  /**
   * @brief Returns the input GPU TensorList with index = idx. 
   */
  const TensorList<GPUBackend>& GPUInputList(int idx);

  /**
   * @brief Returns the output GPU TensorList with index = idx. 
   */
  TensorList<GPUBackend>* GPUOutputList(int idx);

  /**
   * @brief Returns the index of the sample that is to be worked on.
   * If the whole batch is to be computed (gpu ops), the data index 
   * is -1.
   */
  int data_idx() const { return data_idx_; }

  /**
   * @brief Return the index of the thread from the thread pool
   * that is this work will be done by. If the work is not done
   * in the thread pool (gpu ops), the thread index is -1.
   */
  int thread_idx() const { return thread_idx_; }

  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  cudaStream_t stream() const { return stream_; }

  /**
   * @brief Returns the parameter tensor at the given index.
   */
  Tensor<CPUBackend>* ParamTensor(int idx);

  /**
   * @brief Returns the gpu paramter tensor at the given index.
   */
  Tensor<GPUBackend>* GPUParamTensor(int idx);
  
  // Setters for member variables. Used by the Pipeline to
  // setup workspaces for each operator. Should not be
  // touched by Ops.

private:
  vector<TensorList<CPUBackend>*> cpu_inputs_, cpu_outputs_;
  vector<TensorList<GPUBackend>*> gpu_inputs_, gpu_outputs_;

  // Temporary Tensor objects to wrap individual tensors in inputs
  vector<vector<Tensor<CPUBackend>>> cpu_input_wrappers_, cpu_output_wrappers_;
  vector<vector<Tensor<GPUBackend>>> gpu_input_wrappers_, gpu_output_wrappers_;
  
  int data_idx_, thread_idx_;
  cudaStream_t stream_;

  vector<Tensor<CPUBackend>> cpu_parameters_;
  vector<Tensor<GPUBackend>> gpu_parameters_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_WORKSPACE_H_
