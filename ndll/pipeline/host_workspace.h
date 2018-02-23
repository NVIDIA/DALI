// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_HOST_WORKSPACE_H_
#define NDLL_PIPELINE_HOST_WORKSPACE_H_

#include <utility>
#include <vector>
#include <memory>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

class SampleWorkspace;

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
   * @brief Returns a sample workspace for the given sample
   * index and thread index
   */
  void GetSample(SampleWorkspace *ws, int data_idx, int thread_idx);

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
   * @brief Returns the number of Tensors in the output set of
   * tensors at the given index.
   */
  int NumOutputAtIdx(int idx) const;

  /**
   * Returns true if the input Tensors at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx) const;

  /**
   * Returns true if the output Tensors at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx) const;

  /**
   * @brief Returns the Tensor at index `data_idx` in the input
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  const Tensor<Backend>& Input(int idx, int data_idx) const;

  /**
   * @brief Adds the input vector of Tensors as an input
   */
  template <typename Backend>
  void AddInput(vector<shared_ptr<Tensor<Backend>>> input);

  /**
   * @brief Sets the input at the specified index to the input
   * vector of Tensors.
   */
  template <typename Backend>
  void SetInput(int idx, vector<shared_ptr<Tensor<Backend>>> input);

  /**
   * @brief Returns the Tensor at index `data_idx` in the output
   * Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  Tensor<Backend>* Output(int idx, int data_idx);

  /**
   * @brief Returns all output Tensors at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  template <typename Backend>
  vector<shared_ptr<Tensor<Backend>>> SharedOutput(int idx);

  /**
   * @brief Adds the input vector of Tensors as an output
   */
  template <typename Backend>
  void AddOutput(vector<shared_ptr<Tensor<Backend>>> output);

  /**
   * @brief Sets the output at the specified index to the input
   * vector of Tensors.
   */
  template <typename Backend>
  void SetOutput(int idx, vector<shared_ptr<Tensor<Backend>>> output);

 private:
  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  vector<vector<TensorPtr<CPUBackend>>> cpu_inputs_, cpu_outputs_;
  vector<vector<TensorPtr<GPUBackend>>> gpu_inputs_, gpu_outputs_;

  // Maps from a TensorVector position in its typed vector
  // to its absolute position in the workspaces outputs
  vector<int> cpu_inputs_index_, gpu_inputs_index_;
  vector<int> cpu_outputs_index_, gpu_outputs_index_;

  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_HOST_WORKSPACE_H_
