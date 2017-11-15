#ifndef NDLL_PIPELINE_SAMPLE_WORKSPACE_H_
#define NDLL_PIPELINE_SAMPLE_WORKSPACE_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/batch_workspace.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/sample_meta.h"

namespace ndll {

/**
 * @brief SampleWorkspace stores all data required for an operator to
 * perform its computation on a single sample.
 */
class SampleWorkspace {
public:
  /**
   * @brief Construct a Sample workspace from the data at index
   * data_idx in the input Workspace. Save the id of the thread
   * that will process this data.
   */
  SampleWorkspace(BatchWorkspace *ws, int data_idx, int thread_idx) :
    data_idx_(data_idx), thread_idx_(thread_idx) {
    NDLL_FAIL("Not implemented.");
  }
  
  ~SampleWorkspace() = default;

  /**
   * @brief Returns the number of input CPU tensors
   */
  inline int NumInput() { return input_index_map_.size(); }
  
  /**
   * @brief Returns the number of input GPU tensors
   */
  inline int NumOutput() { return output_index_map_.size(); }

  /**
   * Returns true if the input TensorList at the given index has the calling
   * Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx);

  /**
   * Returns true if the output TensorList at the given index has the calling
   * Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx);
  
  /**
   * @brief Returns Tensor with index = data_idx() from the input
   * TensorList at index = `idx`.
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx);

  /**
   * @brief Returns Tensor with index = data_idx() from the output
   * TensorList at index = `idx`.
   */
  template <typename Backend>
  Tensor<Backend>* Output(int idx);

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
   * @brief Updates the internal meta-data object to reflect the current
   * contents of the workspace.
   */
  void UpdateMeta() {
    meta_.SetMeta(this);
  }
  
  /**
   * @brief Returns a SampleMeta object that contains all meta-data for
   * the SampleWorkspace (tensor shapes, types, data_idx, thread_idx, etc.)
   */
  const SampleMeta& meta() const {
    return meta_;
  }
    
  /**
   * @brief Returns the index of the sample that this workspace stores
   * in the input/output batch.
   */
  inline int data_idx() const { return data_idx_; }

  /**
   * @brief Returns the index of the thread that will process this data.
   */
  inline int thread_idx() const { return thread_idx_; }
  
  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const { return stream_; }
  
private:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  vector<TensorPtr<CPUBackend>> cpu_inputs_, cpu_outputs_;
  vector<TensorPtr<GPUBackend>> gpu_inputs_, gpu_outputs_;
  
  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;
  
  int data_idx_, thread_idx_;
  cudaStream_t stream_;
  
  vector<TensorPtr<CPUBackend>> cpu_parameters_;
  vector<TensorPtr<GPUBackend>> gpu_parameters_;

  SampleMeta meta_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_SAMPLE_WORKSPACE_H_ 
