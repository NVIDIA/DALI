// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_WORKSPACE_WORKSPACE_H_
#define DALI_PIPELINE_WORKSPACE_WORKSPACE_H_

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <unordered_map>

#include "dali/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief ArgumentWorskpace is a base class of
 * objects storing tensor arguments
 * of operators
 */
class ArgumentWorkspace {
 public:
  ArgumentWorkspace() {}
  virtual ~ArgumentWorkspace() = default;

  inline void Clear() {
    argument_inputs_.clear();
  }

  void AddArgumentInput(shared_ptr<Tensor<CPUBackend>> input, std::string arg_name) {
    argument_inputs_[arg_name] = input;
  }

  void SetArgumentInput(shared_ptr<Tensor<CPUBackend>> input, std::string arg_name) {
    DALI_ENFORCE(argument_inputs_.find(arg_name) != argument_inputs_.end(),
        "Argument \"" + arg_name + "\" not found.");
    argument_inputs_[arg_name] = input;
  }

  const Tensor<CPUBackend>& ArgumentInput(std::string arg_name) const {
    DALI_ENFORCE(argument_inputs_.find(arg_name) != argument_inputs_.end(),
        "Argument \"" + arg_name + "\" not found.");
    return *(argument_inputs_.at(arg_name));
  }

 protected:
  // Argument inputs
  std::unordered_map<std::string, shared_ptr<Tensor<CPUBackend>>> argument_inputs_;
};

/**
 * @brief WorkspaceBase is a base class of objects
 * storing all data required by an operator,
 * including its input and output, parameter tensors and
 * meta-data about execution.
 */
template <template<typename> class InputType, template<typename> class OutputType>
class WorkspaceBase : public ArgumentWorkspace {
 public:
  template <typename Backend>
  using input_t = InputType<Backend>;

  template <typename Backend>
  using output_t = OutputType<Backend>;

  WorkspaceBase() {}
  ~WorkspaceBase() override = default;

  /**
   * @brief Clears the contents of the workspaces, reseting it
   * to a default state.
   */
  inline void Clear() {
    ArgumentWorkspace::Clear();
    cpu_inputs_.clear();
    gpu_inputs_.clear();
    cpu_outputs_.clear();
    gpu_outputs_.clear();
    input_index_map_.clear();
    output_index_map_.clear();
    cpu_inputs_index_.clear();
    gpu_inputs_index_.clear();
    cpu_outputs_index_.clear();
    gpu_outputs_index_.clear();
  }

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
    DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
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
    DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
    // input_index_map_.first is true if the input is stored on CPU
    // so we do XOR of it with the Backend being GPUBackend
    return output_index_map_[idx].first != std::is_same<Backend, GPUBackend>::value;
  }

  /**
   * @brief Adds new CPU input.
   */
  void AddInput(InputType<CPUBackend> input) {
    AddHelper(input, &cpu_inputs_, &cpu_inputs_index_, &input_index_map_, true);
  }

  /**
   * @brief Adds new GPU input.
   */
  void AddInput(InputType<GPUBackend> input) {
    AddHelper(input, &gpu_inputs_, &gpu_inputs_index_, &input_index_map_, false);
  }

  /**
   * @brief Sets the CPU input at the specified index to the given input argument
   */
  void SetInput(int idx, InputType<CPUBackend> input) {
    SetHelper<InputType, CPUBackend>(idx,
                                     input,
                                     &cpu_inputs_,
                                     &cpu_inputs_index_,
                                     &input_index_map_,
                                     &cpu_inputs_,
                                     &cpu_inputs_index_,
                                     &gpu_inputs_,
                                     &gpu_inputs_index_,
                                     true);
  }

  /**
   * @brief Sets the GPU input at the specified index to the given input argument
   */
  void SetInput(int idx, InputType<GPUBackend> input) {
    SetHelper<InputType, GPUBackend>(idx,
                                     input,
                                     &gpu_inputs_,
                                     &gpu_inputs_index_,
                                     &input_index_map_,
                                     &cpu_inputs_,
                                     &cpu_inputs_index_,
                                     &gpu_inputs_,
                                     &gpu_inputs_index_,
                                     false);
  }

  /**
   * @brief Adds new CPU output
   */
  void AddOutput(OutputType<CPUBackend> output) {
    AddHelper(output, &cpu_outputs_, &cpu_outputs_index_, &output_index_map_, true);
  }

  /**
   * @brief Adds new GPU output
   */
  void AddOutput(OutputType<GPUBackend> output) {
    AddHelper(output, &gpu_outputs_, &gpu_outputs_index_, &output_index_map_, false);
  }

  /**
   * @brief Sets the CPU output at the specified index
   */
  void SetOutput(int idx, OutputType<CPUBackend> output) {
    SetHelper<OutputType, CPUBackend>(idx,
                                      output,
                                      &cpu_outputs_,
                                      &cpu_outputs_index_,
                                      &output_index_map_,
                                      &cpu_outputs_,
                                      &cpu_outputs_index_,
                                      &gpu_outputs_,
                                      &gpu_outputs_index_,
                                      true);
  }

  /**
   * @brief Sets the GPU output at the specified index
   */
  void SetOutput(int idx, OutputType<GPUBackend> output) {
    SetHelper<OutputType, GPUBackend>(idx,
                                      output,
                                      &gpu_outputs_,
                                      &gpu_outputs_index_,
                                      &output_index_map_,
                                      &cpu_outputs_,
                                      &cpu_outputs_index_,
                                      &gpu_outputs_,
                                      &gpu_outputs_index_,
                                      false);
  }

  /**
   * @brief Returns reference to internal CPU output object at index `idx`.
   *
   * @throws runtime_error if the calling type does not match the
   * type of the tensor at the given index
   */
  OutputType<CPUBackend> SharedCPUOutput(int idx) {
    DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
    auto tensor_meta = output_index_map_[idx];
    DALI_ENFORCE(tensor_meta.first, "Output with given "
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
    DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
    auto tensor_meta = output_index_map_[idx];
    DALI_ENFORCE(!tensor_meta.first, "Output with given "
        "index does not have the calling backend type (GPUBackend)");
    return gpu_outputs_[tensor_meta.second];
  }

 protected:
  template <typename T>
  void AddHelper(T entry,
                 vector<T>* vec,
                 vector<int>* index,
                 vector<std::pair<bool, int>>* index_map,
                 const bool on_cpu) {
    // Save the vector of tensors
    vec->push_back(entry);

    // Update the input index map
    index_map->push_back(std::make_pair(on_cpu, vec->size()-1));
    index->push_back(index_map->size()-1);
  }

  template <template<typename> class T, typename Backend>
  void SetHelper(int idx,
                 T<Backend> entry,
                 vector<T<Backend>>* vec,
                 vector<int>* index,
                 vector<std::pair<bool, int>>* index_map,
                 vector<T<CPUBackend>>* cpu_vec,
                 vector<int>* cpu_index,
                 vector<T<GPUBackend>>* gpu_vec,
                 vector<int>* gpu_index,
                 bool on_cpu
                 ) {
    DALI_ENFORCE_VALID_INDEX(idx, index_map->size());

    // To remove the old input at `idx`, we need to remove it
    // from its typed vector and update the index_map
    // entry for all the elements in the vector following it.
    auto tensor_meta = (*index_map)[idx];
    if (tensor_meta.first) {
      for (size_t i = tensor_meta.second; i < cpu_vec->size(); ++i) {
        int &input_idx = (*index_map)[(*cpu_index)[i]].second;
        --input_idx;
      }
      cpu_vec->erase(cpu_vec->begin() + tensor_meta.second);
      cpu_index->erase(cpu_index->begin() + tensor_meta.second);
    } else {
      for (size_t i = tensor_meta.second; i < gpu_vec->size(); ++i) {
        int &input_idx = (*index_map)[(*gpu_index)[i]].second;
        --input_idx;
      }
      gpu_vec->erase(gpu_vec->begin() + tensor_meta.second);
      gpu_index->erase(gpu_index->begin() + tensor_meta.second);
    }

    // Now we insert the new input and update its meta data
    vec->push_back(entry);
    index->push_back(idx);
    (*index_map)[idx] = std::make_pair(on_cpu, vec->size()-1);
  }

  inline const InputType<GPUBackend>& GPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(!tensor_meta.first, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (GPUBackend)");
    return gpu_inputs_[tensor_meta.second];
  }

  inline const InputType<CPUBackend>& CPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(tensor_meta.first, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (CPUBackend)");
    return cpu_inputs_[tensor_meta.second];
  }

  inline const InputType<GPUBackend>& GPUOutput(int idx) const {
    auto tensor_meta = FetchAtIndex(output_index_map_, idx);
    DALI_ENFORCE(!tensor_meta.first, "Output with given "
        "index (" + std::to_string(idx) +
        ")does not have the calling backend type (GPUBackend)");
    return gpu_outputs_[tensor_meta.second];
  }

  inline const InputType<CPUBackend>& CPUOutput(int idx) const {
    auto tensor_meta = FetchAtIndex(output_index_map_, idx);
    DALI_ENFORCE(tensor_meta.first, "Output with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (CPUBackend)");
    return cpu_outputs_[tensor_meta.second];
  }

  vector<InputType<CPUBackend>> cpu_inputs_;
  vector<OutputType<CPUBackend>> cpu_outputs_;
  vector<InputType<GPUBackend>> gpu_inputs_;
  vector<OutputType<GPUBackend>> gpu_outputs_;

  // Maps from a Tensor position in its typed vector
  // to its absolute position in the workspaces outputs
  vector<int> cpu_inputs_index_, gpu_inputs_index_;
  vector<int> cpu_outputs_index_, gpu_outputs_index_;
  // Used to map input/output tensor indices (0, 1, ... , num_input-1)
  // to actual tensor objects. The first element indicates if the
  // Tensor is stored on cpu, and the second element is the index of
  // that tensor in the {cpu, gpu}_inputs_ vector.
  vector<std::pair<bool, int>> input_index_map_, output_index_map_;

 private:
  inline const std::pair<bool, int>& FetchAtIndex(
    const vector<std::pair<bool, int>>& index_map, int idx) const {
    DALI_ENFORCE(idx >= 0 && idx < (int) index_map.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(index_map.size())
      + ")");
    return index_map[idx];
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_H_
