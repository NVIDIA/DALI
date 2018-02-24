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

 protected:

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
