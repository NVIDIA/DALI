#ifndef NDLL_PIPELINE_WORKSPACE_WORKSPACE_H_
#define NDLL_PIPELINE_WORKSPACE_WORKSPACE_H_

#include <vector>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"

namespace ndll {

/**
 * @brief Workspace is a base class of objects
 * storing all data required by an operator,
 * including its input and output, parameter tensors and
 * meta-data about execution.
 */
template <template<typename> class InputType, template<typename> class OutputType>
class Workspace {
 public:
  Workspace() {}
  virtual ~Workspace() = default;

  /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() const { return input_index_map_.size(); }

  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() const { return output_index_map_.size(); }

  /**
   * Returns true if the input at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx) const {
    NDLL_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
    // input_index_map_.first is true if the input is stored on CPU
    // so we do XOR of it with the Backend being GPUBackend
    return input_index_map_[idx].first != std::is_same<Backend, GPUBackend>::value;
  }

  /**
   * Returns true if the output at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx) const {
  NDLL_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
  // input_index_map_.first is true if the input is stored on CPU
  // so we do XOR of it with the Backend being GPUBackend
  return output_index_map_[idx].first != std::is_same<Backend, GPUBackend>::value;
  }

  /**
   * @brief Adds new CPU input.
   */
  void AddInput(InputType<CPUBackend> input) {
    AddInputHelper(input, cpu_inputs_, cpu_inputs_index_);
  }

  /**
   * @brief Adds new GPU input.
   */
  void AddInput(InputType<GPUBackend> input) {
    AddInputHelper(input, gpu_inputs_, gpu_inputs_index_);
  }

  /**
   * @brief Sets the CPU input at the specified index to the given input argument
   */
  void SetInput(int idx, InputType<CPUBackend> input) {
    SetInputHelper(idx, input, cpu_inputs_, cpu_inputs_index_ );
  }

  /**
   * @brief Sets the GPU input at the specified index to the given input argument
   */
  void SetInput(int idx, InputType<GPUBackend> input) {
    SetInputHelper(idx, input, gpu_inputs_, gpu_inputs_index_ );
  }

  /**
   * @brief Returns reference to internal CPU output object at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  OutputType<CPUBackend> SharedCPUOutput(int idx) {
    NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
    auto tensor_meta = output_index_map_[idx];
    NDLL_ENFORCE(tensor_meta.first, "Output with given "
        "index does not have the calling backend type (CPUBackend)");
    return cpu_outputs_[tensor_meta.second];
  }

  /**
   * @brief Returns reference to internal GPU output object at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  OutputType<GPUBackend> SharedGPUOutput(int idx) {
    NDLL_ENFORCE_VALID_INDEX((size_t)idx, output_index_map_.size());
    auto tensor_meta = output_index_map_[idx];
    NDLL_ENFORCE(!tensor_meta.first, "Output with given "
        "index does not have the calling backend type (GPUBackend)");
    return gpu_outputs_[tensor_meta.second];
  }

 protected:

  template <typename Backend>
  void AddInputHelper(InputType<Backend> input,
                      vector<InputType<Backend>>& inputs,
                      vector<int>& inputs_index) {
    // Save the vector of tensors
    inputs.push_back(input);

    // Update the input index map
    input_index_map_.push_back(std::make_pair(true, inputs.size()-1));
    inputs_index.push_back(input_index_map_.size()-1);
  }

  template <typename Backend>
  void SetInputHelper(int idx,
                      InputType<Backend> input,
                      vector<InputType<Backend>>& inputs,
                      vector<int>& inputs_index) {
    NDLL_ENFORCE_VALID_INDEX(idx, input_index_map_.size());

    // To remove the old input at `idx`, we need to remove it
    // from its typed vector and update the input_index_map
    // entry for all the elements in the vector following it.
    auto tensor_meta = input_index_map_[idx];
    if (tensor_meta.first) {
      for (size_t i = tensor_meta.second; i < cpu_inputs_.size(); ++i) {
        int &input_idx = input_index_map_[cpu_inputs_index_[i]].second;
        --input_idx;
      }
      cpu_inputs_.erase(cpu_inputs_.begin() + tensor_meta.second);
      cpu_inputs_index_.erase(cpu_inputs_index_.begin() + tensor_meta.second);
    } else {
      for (size_t i = tensor_meta.second; i < gpu_inputs_.size(); ++i) {
        int &input_idx = input_index_map_[gpu_inputs_index_[i]].second;
        --input_idx;
      }
      gpu_inputs_.erase(gpu_inputs_.begin() + tensor_meta.second);
      gpu_inputs_index_.erase(gpu_inputs_index_.begin() + tensor_meta.second);
    }

    // Now we insert the new input and update its meta data
    inputs.push_back(input);
    inputs_index.push_back(idx);
    input_index_map_[idx] = std::make_pair(true, inputs.size()-1);
  }


  vector<InputType<CPUBackend>> cpu_inputs_;
  vector<OutputType<CPUBackend>> cpu_outputs_;
  vector<InputType<GPUBackend>> gpu_inputs_;
  vector<OutputType<GPUBackend>> gpu_outputs_;

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

#endif  // NDLL_PIPELINE_WORKSPACE_WORKSPACE_H_
