#ifndef NDLL_PIPELINE_HOST_WORKSPACE_H_
#define NDLL_PIPELINE_HOST_WORKSPACE_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

/**
 * @brief HostWorkspace stores all data that a cpu op operates on.
 * HostWorkspace differs from BatchWorkspace in that the input data
 * in a mixed workspace is per-sample, and the outputs are contiguous.
 */
class HostWorkspace {
public:
  inline HostWorkspace() {}
  inline ~HostWorkspace() = default;

    /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() { return input_index_map_.size(); }
  
  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() { return output_index_map_.size(); }

  /**
   * Returns true if the input TensorList at the given index 
   * has the calling Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx);

  /**
   * Returns true if the output TensorList at the given index 
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx);
  
  /**
   * @brief Returns the input TensorList at index `idx`. If the input 
   * at the given index does not match the calling Backend type, this 
   * method throws an error.
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Returns the output Tensor at index `idx`. If the output 
   * at the given index does not match the calling Backend type, this 
   * method throws an error.
   */
  template <typename Backend>
  Tensor<Backend>* Output(int idx, int data_idx);

private:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  vector<vector<TensorPtr<CPUBackend>>> cpu_inputs_, cpu_outputs_;
  vector<vector<TensorPtr<GPUBackend>>> gpu_inputs_, gpu_outputs_;
  
  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;

};

} // namespace ndll

#endif // NDLL_PIPELINE_HOST_WORKSPACE_H_
