// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_MIXED_WORKSPACE_H_
#define NDLL_PIPELINE_MIXED_WORKSPACE_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

class SampleWorkspace;

namespace internal {

/**
 * @brief MixedWorkspace stores all data that an InternalOp operates on.
 * MixedWorkspace differs from BatchWorkspace in that the input data
 * in a mixed workspace is per-sample, and the outputs are contiguous.
 */
class MixedWorkspace {
public:
  inline MixedWorkspace() : stream_(0) {}
  inline ~MixedWorkspace() = default;

  /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() const { return input_index_map_.size(); }

  /**
   * @brief Returns the number of Tensors in the input set of
   * tensors at the given index.
   */
  int NumInputAtIdx(int idx) const;
  
  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() const { return output_index_map_.size(); }

  /**
   * Returns true if the set of input Tensor at the given 
   * index has the calling Backend type.
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
   * @brief Returns the input Tensor at index `data_idx` in the input
   * set of Tensors at index `idx`.
   *
   * @throws runtime_error If calling type does not match the type of 
   * the output at the given index.
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Adds the input vector of Tensors as an input
   */
  template <typename Backend>
  void AddInput(vector<shared_ptr<Tensor<Backend>>> input);
  
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
   * @brief Adds the input TensorList as an output
   */
  template <typename Backend>
  void AddOutput(shared_ptr<TensorList<Backend>> output);

  /**
   * @brief Sets the output at the specified index to the input
   * TensorList
   */
  template <typename Backend>
  void SetOutput(int idx, shared_ptr<TensorList<Backend>> output);
  
  /**
   * @brief Sets the stream for this workspace.
   */
  inline void set_stream(cudaStream_t stream) {
    has_stream_ = true;
    stream_ = stream;
  }

  /**
   * @brief Returns true if 'set_stream' has been called.
   */
  inline bool has_stream() const { return has_stream_; }
  
  /**
   * @brief Returns the cuda stream that this work is to be done in.
   */
  inline cudaStream_t stream() const {
    NDLL_ENFORCE(has_stream_, "Workspace does not have a stream.");
    return stream_;
  }

  /**
   * @brief Sets the event for this workspace.
   */
  inline void set_event(cudaEvent_t event) {
    has_event_ = true;
    event_ = event;
  }

  /**
   * @brief Returns true if 'set_event' has been called.
   */
  inline bool has_event() const { return has_event_; }
  
  /**
   * @brief Returns the cuda event that signals this works completion.
   */
  inline cudaEvent_t event() const {
    NDLL_ENFORCE(has_event_, "Workspace does not have an event.");
    return event_;
  }  

private:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  vector<vector<TensorPtr<CPUBackend>>> cpu_inputs_;
  vector<vector<TensorPtr<GPUBackend>>> gpu_inputs_;
  
  template <typename T>
  using TensorListPtr = shared_ptr<TensorList<T>>;
  vector<TensorListPtr<CPUBackend>> cpu_outputs_;
  vector<TensorListPtr<GPUBackend>> gpu_outputs_;

  // Maps from a TensorLists position in its typed vector
  // to its absolute position in the workspaces outputs
  vector<int> cpu_outputs_index_, gpu_outputs_index_;
  
  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;

  bool has_stream_ = false, has_event_ = false;
  cudaStream_t stream_;
  cudaEvent_t event_;
};

} // namespace internal
} // namespace ndll

#endif // NDLL_PIPELINE_MIXED_WORKSPACE_H_
