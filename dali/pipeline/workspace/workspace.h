// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/workspace/iteration_data.h"

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
    argument_input_idxs_.clear();
    argument_inputs_.clear();
  }

  int NumArgumentInput() const {
    return argument_inputs_.size();
  }

  int AddArgumentInput(std::string arg_name, shared_ptr<TensorList<CPUBackend>> input) {
    int idx = argument_input_idxs_.size();
    argument_input_idxs_[arg_name] = idx;
    argument_inputs_.push_back({ std::move(arg_name), std::move(input) });
    return idx;
  }

  void SetArgumentInput(int idx, shared_ptr<TensorList<CPUBackend>> input) {
    assert(idx >= 0 && idx < static_cast<int>(argument_inputs_.size()));
    argument_inputs_[idx].cpu = std::move(input);
  }

  const TensorList<CPUBackend>& ArgumentInput(std::string_view arg_name) const {
    auto it = argument_input_idxs_.find(arg_name);
    if (it == argument_input_idxs_.end())
      throw invalid_key(make_string("Argument \"", arg_name, "\" not found."));
    assert(argument_inputs_[it->second].cpu);
    return *argument_inputs_[it->second].cpu;
  }

  const std::string &ArgumentInputName(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumArgumentInput());
    return argument_inputs_[idx].name;
  }

  const TensorList<CPUBackend> &ArgumentInput(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumArgumentInput());
    assert(argument_inputs_[idx].cpu);
    return *argument_inputs_[idx].cpu;
  }

  const std::shared_ptr<TensorList<CPUBackend>> ArgumentInputPtr(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumArgumentInput());
    return argument_inputs_[idx].cpu;
  }

  std::shared_ptr<TensorList<CPUBackend>>&
  UnsafeMutableArgumentInput(std::string_view arg_name) {
    auto it = argument_input_idxs_.find(arg_name);
    if (it == argument_input_idxs_.end())
      throw invalid_key(make_string("Argument \"", arg_name, "\" not found."));
    return argument_inputs_[it->second].cpu;
  }

  const auto &ArgumentInputs() const {
    return argument_inputs_;
  }

  struct ArgumentInputBuffers {
    std::string name;
    std::shared_ptr<TensorList<CPUBackend>> cpu;
    // In the future, we may add GPU data here
  };

 protected:
  // Argument inputs
  std::map<std::string, int, std::less<>> argument_input_idxs_;
  SmallVector<ArgumentInputBuffers, 4> argument_inputs_;
};

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
    inputs_.clear();
    outputs_.clear();
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
  const auto& Input(int idx, Backend backend = {}) const {
    return *InputHandle(idx, backend);
  }

  /**
   * @brief Returns the mutable reference to the output batch at the position `idx`.
   *
   * The operator implementation can use this function to access its outputs.
   */
  template <typename Backend>
  auto& Output(int idx, Backend backend = {}) const {
    return *OutputHandle(idx, backend);
  }

  /**
   * @brief Returns the number of inputs.
   */
  inline int NumInput() const {
    return inputs_.size();
  }

  /**
   * @brief Returns the number of outputs.
   */
  inline int NumOutput() const {
    return outputs_.size();
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
  auto& UnsafeMutableInput(int idx, Backend backend = {}) const {
    return *InputHandle(idx, backend);
  }

  /**
   * @brief Returns the underlying handle to the input batch at the position `idx`.
   *
   * Intended only for executor and other internal APIs.
   */
  template <typename Backend>
  const DataObjectPtr<Backend>& InputPtr(int idx, Backend backend = {}) const {
    return InputHandle(idx, backend);
  }

  /**
   * @brief Returns the underlying handle to the output batch at the position `idx`.
   *
   * Intended only for executor and other internal APIs.
   */
  template <typename Backend>
  const DataObjectPtr<Backend>& OutputPtr(int idx, Backend backend = {}) const {
    return OutputHandle(idx, backend);
  }

  /** @} */

  /**
   * Returns shape of the input at given index
   * @return TensorShape<> for SampleWorkspace, TensorListShape<> for other Workspaces
   */
  auto GetInputShape(int input_idx) const {
    return GetBufferProperty(inputs_, input_idx, [](auto &buf) { return buf->shape(); });
  }

  /**
   * Returns shape of the output at given index
   * @return TensorShape<> for SampleWorkspace, TensorListShape<> for other Workspaces
   */
  auto GetOutputShape(int output_idx) const {
    return GetBufferProperty(outputs_, output_idx, [](auto &buf) { return buf->shape(); });
  }

  /**
   * @brief Returns the type of the data in the input at given index.
   */
  DALIDataType GetInputDataType(int input_idx) const {
    return GetBufferProperty(inputs_, input_idx, [](auto &buf) { return buf->type(); });
  }

  /**
   * @brief Returns the type of the data in the output at given index.
   */
  DALIDataType GetOutputDataType(int output_idx) const {
    return GetBufferProperty(outputs_, output_idx, [](auto &buf) { return buf->type(); });
  }

  /**
   * @brief Returns the layout of the input at given index
   */
  TensorLayout GetInputLayout(int input_idx) const {
    return GetBufferProperty(inputs_, input_idx, [](auto &buf) { return buf->GetLayout(); });
  }

  /**
   * @brief Returns the layout of the output at given index
   */
  TensorLayout GetOutputLayout(int output_idx) const {
    return GetBufferProperty(outputs_, output_idx, [](auto &buf) { return buf->GetLayout(); });
  }

  /**
   * Returns batch size for a given input
   */
  int GetInputBatchSize(int input_idx) const {
    return GetBufferProperty(inputs_, input_idx, [](auto &buf) { return buf->num_samples(); });
  }

  /**
   * Returns batch size for a given output
   */
  int GetOutputBatchSize(int output_idx) const {
    return GetBufferProperty(outputs_, output_idx, [](auto &buf) { return buf->num_samples(); });
  }

  /**
   * Returns number of dimensions for a given input
   */
  int GetInputDim(int input_idx) const {
    return GetBufferProperty(inputs_, input_idx, [](auto &buf) { return buf->sample_dim(); });
  }

  /**
   * Returns number of dimensions for a given output
   */
  int GetOutputDim(int output_idx) const {
    return GetBufferProperty(outputs_, output_idx, [](auto &buf) { return buf->sample_dim(); });
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
    DALI_ENFORCE_VALID_INDEX(idx, inputs_.size());
    return inputs_[idx].device == backend_to_storage_device<Backend>::value;
  }

  /**
   * Returns true if the output at the given index
   * has the calling Backend type.
   */
  template <typename Backend>
  bool OutputIsType(int idx) const {
    DALI_ENFORCE_VALID_INDEX(idx, outputs_.size());
    return outputs_[idx].device == backend_to_storage_device<Backend>::value;
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
   * @brief Adds a new input
   *
   * This overload can be useful when there's an automatic conversion, e.g. from a nullptr
   * ws.AddInput<CPUBackend>(nullptr);
   */
  template <typename Backend>
  void AddInput(DataObjectPtr<Backend> input, Backend = {}) {
    AddInput(input);
  }

  /**
   * @brief Adds new CPU input
   */
  void AddInput(DataObjectPtr<CPUBackend> input) {
    inputs_.push_back(IOBuffers{ StorageDevice::CPU, std::move(input), nullptr });
  }

  /**
   * @brief Adds new GPU input
   */
  void AddInput(DataObjectPtr<GPUBackend> input) {
    inputs_.push_back(IOBuffers{ StorageDevice::GPU, nullptr, std::move(input) });
  }


  /**
   * @brief Adds a new output
   *
   * This overload can be useful when there's an automatic conversion, e.g. from a nullptr
   * ws.AddOutput<CPUBackend>(nullptr);
   */
  template <typename Backend>
  void AddOutput(DataObjectPtr<Backend> output, Backend = {}) {
    AddOutput(output);
  }

  /**
   * @brief Adds new CPU output
   */
  void AddOutput(DataObjectPtr<CPUBackend> output) {
    outputs_.push_back(IOBuffers{ StorageDevice::CPU, std::move(output), nullptr });
  }

  /**
   * @brief Adds new GPU output
   */
  void AddOutput(DataObjectPtr<GPUBackend> output) {
    outputs_.push_back(IOBuffers{ StorageDevice::GPU, nullptr, std::move(output) });
  }


  /**
   * @brief Sets an input
   *
   * This overload can be useful when there's an automatic conversion, e.g. from a nullptr
   * ws.SetInput<CPUBackend>(idx, nullptr);
   */
  template <typename Backend>
  void SetInput(int idx, DataObjectPtr<Backend> input, Backend = {}) {
    SetInput(idx, input);
  }

  /**
   * @brief Sets a CPU input
   */
  void SetInput(int idx, DataObjectPtr<CPUBackend> input) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    inputs_[idx] = IOBuffers{ StorageDevice::CPU, std::move(input), nullptr };
  }

  /**
   * @brief Sets a GPU input
   */
  void SetInput(int idx, DataObjectPtr<GPUBackend> input) {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    inputs_[idx] = IOBuffers{ StorageDevice::GPU, nullptr, std::move(input) };
  }


  /**
   * @brief Sets an output
   *
   * This overload can be useful when there's an automatic conversion, e.g. from a nullptr
   * ws.SetOutput<CPUBackend>(idx, nullptr);
   */
  template <typename Backend>
  void SetOutput(int idx, DataObjectPtr<Backend> output, Backend = {}) {
    SetOutput(idx, output);
  }

  /**
   * @brief Sets a CPU output
   */
  void SetOutput(int idx, DataObjectPtr<CPUBackend> output) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    outputs_[idx] = IOBuffers{ StorageDevice::CPU, std::move(output), nullptr };
  }

  /**
   * @brief Sets a GPU output
   */
  void SetOutput(int idx, DataObjectPtr<GPUBackend> output) {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    outputs_[idx] = IOBuffers{ StorageDevice::GPU, nullptr, std::move(output) };
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
  /** Sets shared data associated with the current iteration */
  void InjectIterationData(SharedIterData iter_data) {
    if (iter_data != iter_data_) {
      operator_traces_ = nullptr;
      iter_data_ = std::move(iter_data);
    }
  }

  /** Gets the shared data associated with the current iteration */
  SharedIterData GetIterationData() const {
    return iter_data_;
  }

  /**
   * Set the trace value for the current operator.
   *
   * Typically, this function shall be called by an operator in RunImpl or SetupImpl.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void SetOperatorTrace(std::string trace_key, std::string trace_value) {
    GetOperatorTraces().insert_or_assign(std::move(trace_key), std::move(trace_value));
  }

  /**
   * Erase the trace value for the current operator.
   *
   * Typically, this function shall be called by an operator in RunImpl or SetupImpl.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void EraseOperatorTrace(std::string_view trace_key) {
    GetOperatorTraces().erase(trace_key);
  }


  /**
   * Erase all the trace values for the current operator.
   *
   * @see operator_trace_map_t
   */
  DLL_PUBLIC void ClearOperatorTraces() {
    GetOperatorTraces().clear();
  }

  /** Gets operator traces for the currently assigned operator.
   *
   * Returns a map, that maps a trace key to a trace value: `ret_value[trace_key] = trace_value`.
   * The usage is typically within the scope of an operator.
   */
  DLL_PUBLIC auto &GetOperatorTraces() const {
    if (!operator_traces_)
      operator_traces_ = &iter_data_->operator_traces.Get(operator_instance_name_);
    return *operator_traces_;
  }

  /** Get the trace map for a given operator.
   *
   * Returns a map, that maps a trace key to a trace value: `ret_value[trace_key] = trace_value`.
   *
   * Typically, this function is be called when the traces are read.
   *
   * @see operator_trace_map_t
   *
   * @param operator_name Name (ID) of the operator.
   */
  DLL_PUBLIC auto &GetOperatorTraces(std::string_view operator_name) const {
    return iter_data_->operator_traces.Get(operator_name);
  }
  ///@}


 protected:
  template <typename Backend>
  const DataObjectPtr<Backend>& InputHandle(int idx) const {
    return InputHandle(idx, Backend{});
  }

  template <typename Backend>
  const DataObjectPtr<Backend>& OutputHandle(int idx) const {
    return OutputHandle(idx, Backend{});
  }

  const DataObjectPtr<CPUBackend>& InputHandle(int idx, const CPUBackend&) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    DALI_ENFORCE(inputs_[idx].device == StorageDevice::CPU,
      make_string("The input ", idx, " is not on the requested device (CPU)."));
    return inputs_[idx].cpu;
  }

  const DataObjectPtr<GPUBackend>& InputHandle(int idx, const GPUBackend&) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumInput());
    DALI_ENFORCE(inputs_[idx].device == StorageDevice::GPU,
      make_string("The input ", idx, " is not on the requested device (GPU)."));
    return inputs_[idx].gpu;
  }

  const DataObjectPtr<CPUBackend>& OutputHandle(int idx, const CPUBackend&) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    DALI_ENFORCE(outputs_[idx].device == StorageDevice::CPU,
      make_string("The output ", idx, " is not on the requested device (CPU)."));
    return outputs_[idx].cpu;
  }

  const DataObjectPtr<GPUBackend>& OutputHandle(int idx, const GPUBackend&) const {
    DALI_ENFORCE_VALID_INDEX(idx, NumOutput());
    DALI_ENFORCE(outputs_[idx].device == StorageDevice::GPU,
      make_string("The output ", idx, " is not on the requested device (GPU)."));
    return outputs_[idx].gpu;
  }

  /**
   * Batch sizes for given input indices
   */
  SmallVector<int, 4> batch_sizes_;

  struct IOBuffers {
    StorageDevice device;
    DataObjectPtr<CPUBackend> cpu;
    DataObjectPtr<GPUBackend> gpu;

    template <typename Backend>
    DataObjectPtr<Backend> &get(Backend = {}) {
      static_assert(std::is_same_v<Backend, CPUBackend> || std::is_same_v<Backend, GPUBackend>);
      if constexpr (std::is_same_v<Backend, CPUBackend>) {
        DALI_ENFORCE(device == StorageDevice::CPU);
        return cpu;
      } else {
        DALI_ENFORCE(device == StorageDevice::GPU);
        return gpu;
      }
    }
  };

  SmallVector<IOBuffers, 4> inputs_;
  SmallVector<IOBuffers, 2> outputs_;

 private:
  template <typename Buffers, typename Getter>
  static auto GetBufferProperty(Buffers &buffers, int idx, Getter &&getter) {
    DALI_ENFORCE_VALID_INDEX(idx, buffers.size());
    auto &inp = buffers[idx];
    if (inp.device == StorageDevice::GPU)
      return getter(inp.gpu);
    else
      return getter(inp.cpu);
  }

  AccessOrder output_order_ = AccessOrder::host();
  ThreadPool *thread_pool_ = nullptr;
  cudaEvent_t event_ = nullptr;
  SmallVector<cudaEvent_t, 4> parent_events_;

  /** Name of the instance of the operator which this Workspace in associated with. */
  std::string operator_instance_name_;

  /** Cached pointer to the traces for the operator which this Workspace is associated with.
   *
   * mutable, because it's an access accelerator, not a true data member.
   */
  mutable operator_trace_map_t *operator_traces_ = nullptr;

  /** Data shared across all workspaces in the current iteration. */
  SharedIterData iter_data_;
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
