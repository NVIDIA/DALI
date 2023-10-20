// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/executor/iteration_data.h"

namespace dali {

class ThreadPool;

/**
 * @brief Used to specify the shape and type of Output
 * that a Workspace can hold.
 */
struct OutputDesc {
  TensorListShape<> shape;
  DALIDataType type;
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

  void AddArgumentInput(const std::string& arg_name, shared_ptr<TensorList<CPUBackend>> input) {
    argument_inputs_[arg_name] = { std::move(input) };
  }

  const TensorList<CPUBackend>& ArgumentInput(const std::string& arg_name) const {
    auto it = argument_inputs_.find(arg_name);
    DALI_ENFORCE(it != argument_inputs_.end(), "Argument \"" + arg_name + "\" not found.");
    return *it->second.tvec;
  }

  TensorList<CPUBackend>& UnsafeMutableArgumentInput(const std::string& arg_name) {
    return const_cast<TensorList<CPUBackend>&>(ArgumentInput(arg_name));
  }

 protected:
  struct ArgumentInputDesc {
    std::shared_ptr<TensorList<CPUBackend>> tvec;
  };

  // Argument inputs
  using argument_input_storage_t = std::unordered_map<std::string, ArgumentInputDesc>;
  argument_input_storage_t argument_inputs_;

 public:
  using const_iterator = argument_input_storage_t::const_iterator;
  friend const_iterator begin(const ArgumentWorkspace&);
  friend const_iterator end(const ArgumentWorkspace&);
};

/** @{ */
/**
 * @brief Iterator-handling functions for ArgumentWorkspace
 */
inline ArgumentWorkspace::const_iterator begin(const ArgumentWorkspace& ws) {
  return ws.argument_inputs_.begin();
}

inline ArgumentWorkspace::const_iterator end(const ArgumentWorkspace& ws) {
  return ws.argument_inputs_.end();
}
/** @} */

/**
 * @brief WorkspaceBase is a base class of objects
 * storing all data required by an operator,
 * including its input and output, parameter tensors and
 * meta-data about execution.
 *
 * @tparam DataObject - the class template used for storing data (`TensorList` or `Tensor`)
 * @tparam ptr_t      - the template that creates a pointer from the `DataObject<Backend>`;
 *                      currently `std::shared_ptr` for `Workspace`
 *                      and `std::add_pointer_t` for `SampleWorkspace`
 */
template <template <typename Backend> class DataObject,
          template <typename T> class ptr_t = std::shared_ptr>
class WorkspaceBase : public ArgumentWorkspace {
 public:
  template <typename Backend>
  using DataObjectPtr = ptr_t<DataObject<Backend>>;

  WorkspaceBase() = default;
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

  /** @name Input and output APIs
   * Functions used to access inputs and outputs of the operator in its implementation.
   * The inputs are read-only while outputs can be modified.
   * @{
   */

  /**
   * @brief Returns the const reference to the input batch at the position `idx`.
   *
   * The operator implementation can use this function to access its inputs.
   */
  template <typename Backend>
  const auto& Input(int idx) const {
    return *InputHandle(idx, Backend{});
  }

  /**
   * @brief Returns the mutable reference to the output batch at the position `idx`.
   *
   * The operator implementation can use this function to access its outputs.
   */
  template <typename Backend>
  auto& Output(int idx) const {
    return *OutputHandle(idx, Backend{});
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


  /** @} */

  /** @name Internal API for input and output access
   * Functions allowing mutable access to both inputs and outputs that should not be used in
   * operator implementation.
   * @{
   */

  /**
   * @brief Returns the mutable reference to the input batch at the position `idx`.
   *
   * Intended only for executor and other internal APIs.
   */
  template <typename Backend>
  auto& UnsafeMutableInput(int idx) const {
    return *InputHandle(idx, Backend{});
  }

  /**
   * @brief Returns the underlying handle to the input batch at the position `idx`.
   *
   * Intended only for executor and other internal APIs.
   */
  template <typename Backend>
  const DataObjectPtr<Backend>& InputPtr(int idx) const {
    return InputHandle(idx, Backend{});
  }

  /**
   * @brief Returns the underlying handle to the output batch at the position `idx`.
   *
   * Intended only for executor and other internal APIs.
   */
  template <typename Backend>
  const DataObjectPtr<Backend>& OutputPtr(int idx) const {
    return OutputHandle(idx, Backend{});
  }

  /** @} */

  /**
   * Returns shape of input at given index
   * @return TensorShape<> for SampleWorkspace, TensorListShape<> for other Workspaces
   */
  auto GetInputShape(int input_idx) const {
    if (InputIsType<GPUBackend>(input_idx)) {
      return Input<GPUBackend>(input_idx).shape();
    } else {
      return Input<CPUBackend>(input_idx).shape();
    }
  }

  /**
   * Returns the data type of the input at given index
   * @return DALIDataType
   */
  DALIDataType GetInputDataType(int input_idx) const {
    if (InputIsType<GPUBackend>(input_idx)) {
      return Input<GPUBackend>(input_idx).type();
    } else {
      return Input<CPUBackend>(input_idx).type();
    }
  }

  /**
   * @return Type of the data in the output with given index.
   */
  DALIDataType GetOutputDataType(int output_idx) const {
    DALI_ENFORCE(NumOutput() > 0, "No outputs found");
    DALI_ENFORCE(
        output_idx >= 0 && output_idx < NumOutput(),
        make_string("Invalid output index: ", output_idx, "; while NumOutput: ", NumOutput()));
    if (OutputIsType<GPUBackend>(output_idx)) {
      return Output<GPUBackend>(output_idx).type();
    } else {
      return Output<CPUBackend>(output_idx).type();
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
      return Input<GPUBackend>(input_idx).num_samples();
    } else {
      return Input<CPUBackend>(input_idx).num_samples();
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
      return Input<GPUBackend>(input_idx).sample_dim();
    } else {
      return Input<CPUBackend>(input_idx).sample_dim();
    }
  }

  /**
   * Returns number of dimensions for a given output
   */
  int GetOutputDim(int output_idx) const {
    DALI_ENFORCE(NumOutput() > 0, "No outputs found");
    DALI_ENFORCE(
        output_idx >= 0 && output_idx < NumOutput(),
        make_string("Invalid output index: ", output_idx, "; while NumOutput: ", NumOutput()));
    if (OutputIsType<GPUBackend>(output_idx)) {
      return Output<GPUBackend>(output_idx).sample_dim();
    } else {
      return Output<CPUBackend>(output_idx).sample_dim();
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
  void AddInput(DataObjectPtr<CPUBackend> input) {
    AddHelper(input, &cpu_inputs_, &cpu_inputs_index_, &input_index_map_, StorageDevice::CPU);
  }

  /**
   * @brief Adds new GPU input.
   */
  void AddInput(DataObjectPtr<GPUBackend> input) {
    AddHelper(input, &gpu_inputs_, &gpu_inputs_index_, &input_index_map_, StorageDevice::GPU);
  }

  /**
   * @brief Sets the CPU input at the specified index to the given input argument
   */
  void SetInput(int idx, DataObjectPtr<CPUBackend> input) {
    SetHelper<DataObjectPtr, CPUBackend>(
      idx,
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
  void SetInput(int idx, DataObjectPtr<GPUBackend> input) {
    SetHelper<DataObjectPtr, GPUBackend>(
      idx,
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

  inline void SetThreadPool(ThreadPool *pool) {
    thread_pool_ = pool;
  }

  inline bool HasThreadPool() const {
    return thread_pool_ != nullptr;
  }

  inline ThreadPool &GetThreadPool() const {
    DALI_ENFORCE(HasThreadPool(), "Workspace does not have a Thread Pool.");
    return *thread_pool_;
  }

  /**
   * @brief Returns true if this workspace has CUDA stream available
   */
  inline bool has_stream() const {
    return output_order_.is_device();
  }


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
    return output_order_.stream();
  }

  void set_output_order(AccessOrder order) {
    output_order_ = order;
  }

  AccessOrder output_order() const {
    return output_order_;
  }

  void set_stream(cudaStream_t stream) {
    output_order_ = stream;
  }

  inline void set_event(cudaEvent_t event) {
    event_ = event;
  }

  inline cudaEvent_t event() const {
    return event_;
  }

  inline bool has_event() const {
    return event_ != nullptr;
  }

  /**
   * @brief Adds a parent event that will signal this
   * work is allowed to execute.
   */
  inline void AddParentEvent(cudaEvent_t event) { parent_events_.push_back(event); }

  /**
   * @brief Returns the set of parent events this workspace stores.
   */
  inline span<const cudaEvent_t> ParentEvents() const { return make_cspan(parent_events_); }

  /**
   * @brief Adds new CPU output
   */
  void AddOutput(DataObjectPtr<CPUBackend> output) {
    AddHelper(output, &cpu_outputs_, &cpu_outputs_index_, &output_index_map_, StorageDevice::CPU);
  }

  /**
   * @brief Adds new GPU output
   */
  void AddOutput(DataObjectPtr<GPUBackend> output) {
    AddHelper(output, &gpu_outputs_, &gpu_outputs_index_, &output_index_map_, StorageDevice::GPU);
  }

  /**
   * @brief Sets the CPU output at the specified index
   */
  void SetOutput(int idx, DataObjectPtr<CPUBackend> output) {
    SetHelper<DataObjectPtr, CPUBackend>(
      idx,
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
  void SetOutput(int idx, DataObjectPtr<GPUBackend> output) {
    SetHelper<DataObjectPtr, GPUBackend>(
      idx,
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
  DataObjectPtr<CPUBackend> SharedCPUOutput(int idx) {
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
  DataObjectPtr<GPUBackend> SharedGPUOutput(int idx) {
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


  /// @{
  /**
   * Sets the operator ID that this Workspace in associated with.
   */
  void SetOperatorInstanceName(std::string operator_id) {
    operator_instance_name_ = std::move(operator_id);
  }


  /**
   * Returns the operator ID that this Workspace in associated with.
   *
   * @remark When implementing the error messages within an operator implementation,
   * it is not necessary to add the OperatorId to the message - the Executor does it automatically.
   */
  const std::string &GetOperatorInstanceName() const {
    return operator_instance_name_;
  }
  /// @}


  ///@{
  /**
   * Set the whole operator trace map in this workspace.
   *
   * Sets the operator trace map that corresponds to all operators in the current iteration.
   *
   * Typically, this function shall be called by the Executor, when assigning the Workspace to
   * the Operator.
   */
  void InjectOperatorTraces(std::shared_ptr<operator_trace_map_t> operator_trace_map) {
    operator_traces_ = std::move(operator_trace_map);
  }


  /**
   * Set the trace value for the current operator.
   *
   * Typically, this function shall be called by an operator in RunImpl or SetupImpl.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void SetOperatorTrace(const std::string &trace_key, std::string trace_value) {
    (*operator_traces_)[GetOperatorInstanceName()].insert_or_assign(
            trace_key, std::move(trace_value));
  }


  /**
   * Erase the trace value for the current operator.
   *
   * Typically, this function shall be called by an operator in RunImpl or SetupImpl.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void EraseOperatorTrace(const std::string &trace_key) {
    (*operator_traces_)[GetOperatorInstanceName()].erase(trace_key);
  }


  /**
   * Erase all the trace values for the current operator.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void ClearOperatorTraces() {
    (*operator_traces_)[GetOperatorInstanceName()].clear();
  }


  /**
   * Get the trace map for a given operator.
   *
   * Returns a map, that maps a trace key to a trace value: `ret_value[trace_key] = trace_value`.
   *
   * Typically, this function will be called when the traces shall be read.
   *
   * @see operator_trace_map_t
   *
   * @param operator_name Name (ID) of the operator.
   */
  DLL_PUBLIC const auto &GetOperatorTraces(const std::string &operator_name) const {
    return operator_traces_->at(operator_name);
  }

  /**
   * Get the operator trace map for all operators in the pipeline.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC const auto &GetOperatorTraceMap() const {
    return *operator_traces_;
  }
  ///@}


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


  const DataObjectPtr<CPUBackend>& InputHandle(int idx, const CPUBackend&) const {
    return CPUInput(idx);
  }

  const DataObjectPtr<GPUBackend>& InputHandle(int idx, const GPUBackend&) const {
    return GPUInput(idx);
  }

  const DataObjectPtr<CPUBackend>& OutputHandle(int idx, const CPUBackend&) const {
    return CPUOutput(idx);
  }

  const DataObjectPtr<GPUBackend>& OutputHandle(int idx, const GPUBackend&) const {
    return GPUOutput(idx);
  }

  inline const DataObjectPtr<GPUBackend>& GPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::GPU, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (GPUBackend)");
    return gpu_inputs_[tensor_meta.index];
  }

  inline const DataObjectPtr<CPUBackend>& CPUInput(int idx) const {
    auto tensor_meta = FetchAtIndex(input_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::CPU, "Input with given "
        "index (" + std::to_string(idx) +
        ") does not have the calling backend type (CPUBackend)");
    return cpu_inputs_[tensor_meta.index];
  }

  inline const DataObjectPtr<GPUBackend>& GPUOutput(int idx) const {
    auto tensor_meta = FetchAtIndex(output_index_map_, idx);
    DALI_ENFORCE(tensor_meta.storage_device == StorageDevice::GPU,
                 make_string("Output with given index (", idx,
                             ") does not have the calling backend type (GPUBackend)"));
    return gpu_outputs_[tensor_meta.index];
  }

  inline const DataObjectPtr<CPUBackend>& CPUOutput(int idx) const {
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

  vector<DataObjectPtr<CPUBackend>> cpu_inputs_;
  vector<DataObjectPtr<CPUBackend>> cpu_outputs_;
  vector<DataObjectPtr<GPUBackend>> gpu_inputs_;
  vector<DataObjectPtr<GPUBackend>> gpu_outputs_;

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


  AccessOrder output_order_ = AccessOrder::host();
  ThreadPool *thread_pool_ = nullptr;
  cudaEvent_t event_ = nullptr;
  SmallVector<cudaEvent_t, 4> parent_events_;

  /// Name of the instance of the operator, to which this Workspace in associated with.
  std::string operator_instance_name_;

  /// Traces of the operators corresponding to all operators in the current iteration.
  std::shared_ptr<operator_trace_map_t> operator_traces_;
};

class Workspace : public WorkspaceBase<TensorList> {};

class SampleWorkspace;

template <typename Backend>
struct LegacyWorkspace {
  using type = Workspace;
};

template <>
struct LegacyWorkspace<CPUBackend> {
  using type = SampleWorkspace;
};

template <typename Backend>
using legacy_workspace_t = typename LegacyWorkspace<Backend>::type;

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_H_
