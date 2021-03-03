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

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_vector.h"

namespace dali {

/**
 * @brief Used to specify the shape and type of Output
 * that a Workspace can hold.
 */
struct OutputDesc {
  TensorListShape<> shape;
  TypeInfo type;
};

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

  void AddArgumentInput(const std::string &arg_name, shared_ptr<TensorVector<CPUBackend>> input) {
    argument_inputs_[arg_name] = { std::move(input), false };
  }

  void AddArgumentInput(const std::string &arg_name, shared_ptr<TensorList<CPUBackend>> input) {
    argument_inputs_[arg_name] = {
      std::make_shared<TensorVector<CPUBackend>>(std::move(input)),
      true
    };
  }

  const TensorVector<CPUBackend>& ArgumentInput(const std::string &arg_name) const {
    auto it = argument_inputs_.find(arg_name);
    DALI_ENFORCE(it != argument_inputs_.end(), "Argument \"" + arg_name + "\" not found.");
    if (it->second.should_update) {
      // the underlying tensor list might have changed - update the views
      it->second.tvec->UpdateViews();
    }
    return *it->second.tvec;
  }

 protected:
  struct ArgumentInputDesc {
    shared_ptr<TensorVector<CPUBackend>> tvec;
    // If true, the views in TensorVector are updated to reflect the underlying TensorList;
    // this only happens if AddArgumentInput is called with a TensorList pointer - which for now
    // is only when passing an argument input to a GPU stage when using separated queue policy
    // (see queue_policy.h and pipeline.cc for details).
    bool should_update = false;
  };

  // Argument inputs
  using argument_input_storage_t = std::unordered_map<std::string, ArgumentInputDesc>;
  argument_input_storage_t argument_inputs_;

 public:
  using const_iterator = argument_input_storage_t::const_iterator;
  friend const_iterator begin(const ArgumentWorkspace&);
  friend const_iterator end(const ArgumentWorkspace&);
};

/// @{
/**
 * Iterator-handling functions for ArgumentWorkspace
 */
inline ArgumentWorkspace::const_iterator begin(const ArgumentWorkspace& ws) {
  return ws.argument_inputs_.begin();
}

inline ArgumentWorkspace::const_iterator end(const ArgumentWorkspace& ws) {
  return ws.argument_inputs_.end();
}
/// @}

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

  template <typename Backend>
  typename InputType<Backend>::element_type& InputRef(int idx) const {
    return *InputHandle(idx, Backend{});
  }

  template <typename Backend>
  typename OutputType<Backend>::element_type& OutputRef(int idx) const {
    return *OutputHandle(idx, Backend{});
  }

  template <typename Backend>
  const InputType<Backend>& InputPtr(int idx) const {
    return InputHandle(idx, Backend{});
  }

  template <typename Backend>
  const OutputType<Backend>& OutputPtr(int idx) const {
    return OutputHandle(idx, Backend{});
  }

  /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() const {
    return input_index_map_.size();
  }

  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() const {
    return output_index_map_.size();
  }

  /**
   * Returns shape of input at given index
   * @return TensorShape<> for SampleWorkspace, TensorListShape<> for other Workspaces
   */
  auto GetInputShape(int input_idx) const {
    if (InputIsType<GPUBackend>(input_idx)) {
      return InputRef<GPUBackend>(input_idx).shape();
    } else {
      return InputRef<CPUBackend>(input_idx).shape();
    }
  }

  /**
   * Returns batch size for a given input
   */
  int GetInputBatchSize(int input_idx) const {
    DALI_ENFORCE(NumInput() > 0, "No inputs found");
    DALI_ENFORCE(input_idx >= 0 && input_idx < NumInput(),
                 make_string("Invalid input index: ", input_idx, "; while NumInput: ", NumInput()));
    if (InputIsType<GPUBackend>(input_idx)) {
      return InputRef<GPUBackend>(input_idx).ntensor();
    } else {
      return InputRef<CPUBackend>(input_idx).ntensor();
    }
  }

  /**
   * Returns number of dimensions for a given input
   */
  int GetInputDim(int input_idx) const {
    DALI_ENFORCE(NumInput() > 0, "No inputs found");
    DALI_ENFORCE(input_idx >= 0 && input_idx < NumInput(),
                 make_string("Invalid input index: ", input_idx, "; while NumInput: ", NumInput()));
    if (InputIsType<GPUBackend>(input_idx)) {
      return InputRef<GPUBackend>(input_idx).sample_dim();
    } else {
      return InputRef<CPUBackend>(input_idx).sample_dim();
    }
  }

  /**
   * Returns batch size that the Operator is expected to produce on a given output
   */
  int GetRequestedBatchSize(int output_idx) const {
    return batch_sizes_[output_idx];
  }

  /**
   * Set requested batch size for all outputs
   */
  void SetBatchSizes(int batch_size) {
    batch_sizes_.clear();
    batch_sizes_.resize(NumOutput(), batch_size);
  }

  /**
   * Returns true if the input at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool InputIsType(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, input_index_map_.size());
    return input_index_map_[idx].storage_device == backend_to_storage_device<Backend>::value;
  }

  /**
   * Returns true if the output at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, output_index_map_.size());
    return output_index_map_[idx].storage_device == backend_to_storage_device<Backend>::value;
  }

  /**
   * @brief Adds new CPU input.
   */
  void AddInput(InputType<CPUBackend> input) {
    AddHelper(input, &cpu_inputs_, &cpu_inputs_index_, &input_index_map_, StorageDevice::CPU);
  }

  /**
   * @brief Adds new GPU input.
   */
  void AddInput(InputType<GPUBackend> input) {
    AddHelper(input, &gpu_inputs_, &gpu_inputs_index_, &input_index_map_, StorageDevice::GPU);
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
                                     StorageDevice::CPU);
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
                                     StorageDevice::GPU);
  }


  /**
   * @brief Returns true if this workspace has CUDA stream available
   */
  virtual bool has_stream() const = 0;


  /**
   * @brief Returns the CUDA stream that this work is to be done in.
   */
  cudaStream_t stream() const {
    DALI_ENFORCE(has_stream(),
                 "No valid CUDA stream in the Workspace. "
                 "Either the Workspace doesn't support CUDA streams or "
                 "the stream hasn't been successfully set. "
                 "Use `has_stream()`, to runtime-check, "
                 "if CUDA stream is available for this workspace");
    auto stream = stream_impl();
    return stream;
  }


  /**
   * @brief Adds new CPU output
   */
  void AddOutput(OutputType<CPUBackend> output) {
    AddHelper(output, &cpu_outputs_, &cpu_outputs_index_, &output_index_map_, StorageDevice::CPU);
  }

  /**
   * @brief Adds new GPU output
   */
  void AddOutput(OutputType<GPUBackend> output) {
    AddHelper(output, &gpu_outputs_, &gpu_outputs_index_, &output_index_map_, StorageDevice::GPU);
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
                                      StorageDevice::CPU);
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
                                      StorageDevice::GPU);
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
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::CPU, "Output with given "
        "index does not have the calling backend type (CPUBackend)");
    return cpu_outputs_[tensor_meta.index];
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
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::GPU, "Output with given "
        "index does not have the calling backend type (GPUBackend)");
    return gpu_outputs_[tensor_meta.index];
  }

  /**
 * @brief Returns the index of the sample that this workspace stores
 * in the input/output batch.
 */
  DLL_PUBLIC virtual inline int data_idx() const {
    return 0;
  }

  /**
 * @brief Returns the index of the thread that will process this data.
 */
  DLL_PUBLIC virtual inline int thread_idx() const {
    return 0;
  }

 protected:
  struct InOutMeta {
    // Storage device of given Input/Output
    StorageDevice storage_device;
    // Position in dedicated buffer for given storage_device
    int index;

    InOutMeta() : storage_device(static_cast<StorageDevice>(-1)), index(-1) {}
    InOutMeta(StorageDevice storage_device, int index)
        : storage_device(storage_device), index(index) {}
  };

  template <typename T>
  void AddHelper(T entry,
                 vector<T>* vec,
                 vector<int>* index,
                 vector<InOutMeta>* index_map,
                 StorageDevice storage_device) {
    // Save the vector of tensors
    vec->push_back(entry);

    // Update the input index map
    index_map->emplace_back(storage_device, vec->size()-1);
    index->push_back(index_map->size()-1);
  }

  template <template<typename> class T, typename Backend>
  void SetHelper(int idx,
                 T<Backend> entry,
                 vector<T<Backend>>* vec,
                 vector<int>* index,
                 vector<InOutMeta>* index_map,
                 vector<T<CPUBackend>>* cpu_vec,
                 vector<int>* cpu_index,
                 vector<T<GPUBackend>>* gpu_vec,
                 vector<int>* gpu_index,
                 StorageDevice storage_device
                 ) {
    DALI_ENFORCE_VALID_INDEX(idx, index_map->size());

    // To remove the old input at `idx`, we need to remove it
    // from its typed vector and update the index_map
    // entry for all the elements in the vector following it.
    auto tensor_meta = (*index_map)[idx];
    if (tensor_meta.storage_device == StorageDevice::CPU) {
      for (size_t i = tensor_meta.index; i < cpu_vec->size(); ++i) {
        int &input_idx = (*index_map)[(*cpu_index)[i]].index;
        --input_idx;
      }
      cpu_vec->erase(cpu_vec->begin() + tensor_meta.index);
      cpu_index->erase(cpu_index->begin() + tensor_meta.index);
    } else {
      for (size_t i = tensor_meta.index; i < gpu_vec->size(); ++i) {
        int &input_idx = (*index_map)[(*gpu_index)[i]].index;
        --input_idx;
      }
      gpu_vec->erase(gpu_vec->begin() + tensor_meta.index);
      gpu_index->erase(gpu_index->begin() + tensor_meta.index);
    }

    // Now we insert the new input and update its meta data
    vec->push_back(entry);
    index->push_back(idx);
    (*index_map)[idx] = InOutMeta(storage_device, vec->size()-1);
  }


  const InputType<CPUBackend>& InputHandle(int idx, const CPUBackend&) const {
    return CPUInput(idx);
  }

  const InputType<GPUBackend>& InputHandle(int idx, const GPUBackend&) const {
    return GPUInput(idx);
  }

  const OutputType<CPUBackend>& OutputHandle(int idx, const CPUBackend&) const {
    return CPUOutput(idx);
  }

  const OutputType<GPUBackend>& OutputHandle(int idx, const GPUBackend&) const {
    return GPUOutput(idx);
  }

  inline const InputType<GPUBackend>& GPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::GPU, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (GPUBackend)");
    return gpu_inputs_[tensor_meta.index];
  }

  inline const InputType<CPUBackend>& CPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::CPU, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (CPUBackend)");
    return cpu_inputs_[tensor_meta.index];
  }

  inline const OutputType<GPUBackend>& GPUOutput(int idx) const {
    auto tensor_meta = FetchAtIndex(output_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::GPU,
                 make_string("Output with given index (", idx,
                             ") does not have the calling backend type (GPUBackend)"));
    return gpu_outputs_[tensor_meta.index];
  }

  inline const OutputType<CPUBackend>& CPUOutput(int idx) const {
    auto tensor_meta = FetchAtIndex(output_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::CPU,
                 make_string("Output with given index (", idx,
                             ") does not have the calling backend type (CPUBackend)"));
    return cpu_outputs_[tensor_meta.index];
  }

  /**
   * Batch sizes for given input indices
   */
  SmallVector<int, 4> batch_sizes_;

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
  vector<InOutMeta> input_index_map_, output_index_map_;

 private:
  inline const InOutMeta& FetchAtIndex(const vector<InOutMeta>& index_map, int idx) const {
    DALI_ENFORCE(idx >= 0 && idx < (int) index_map.size(),
      "Index out of range." + std::to_string(idx) +
      " not in range [0, " + std::to_string(index_map.size())
      + ")");
    return index_map[idx];
  }


  /**
   * @brief Returns CUDA stream or nullptr, if the stream is unavailable
   */
  virtual cudaStream_t stream_impl() const = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_H_
