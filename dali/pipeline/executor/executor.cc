// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <condition_variable>
#include <iterator>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/executor/source_info_propagation.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/operator/builtin/conditional/split_merge.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

namespace {

bool HasTensorArgInputs(const ArgumentWorkspace& argument_ws) {
  return begin(argument_ws) != end(argument_ws);
}

/**
 * @brief Returns true if all the inputs (regular and argument inputs) held in this ws are empty.
 */
bool AllInputsEmpty(const Workspace &ws) {
  bool all_inputs_empty = true;
  for (int i = 0; i < ws.NumInput(); i++) {
    all_inputs_empty = all_inputs_empty && ws.GetInputBatchSize(i) == 0;
  }
  const ArgumentWorkspace &argument_ws = ws;
  for (const auto &[name, arg] : argument_ws) {
    all_inputs_empty = all_inputs_empty && arg.tvec->num_samples() == 0;
  }
  return all_inputs_empty;
}

/**
 * @brief Returns true if any of the inputs (regular and argument inputs) or the requested output
 * for this workspace are smaller than the max batch size indicating partial batch being processed
 * in a conditional scope (or due to variable BS).
 */
bool AnyBatchPartial(const Workspace &ws, const OpSpec &spec, int max_batch_size) {
  bool any_batch_partial = false;
  if (ws.NumInput() > 0 || HasTensorArgInputs(ws)) {
    for (int i = 0; i < ws.NumInput(); i++) {
      any_batch_partial = any_batch_partial || ws.GetInputBatchSize(i) < max_batch_size;
    }
    const ArgumentWorkspace &argument_ws = ws;
    for (const auto &[name, arg] : argument_ws) {
      any_batch_partial = any_batch_partial || arg.tvec->num_samples() < max_batch_size;
    }
  }
  for (int i = 0; i < spec.NumOutput(); i++) {
    any_batch_partial = any_batch_partial || ws.GetRequestedBatchSize(i) < max_batch_size;
  }
  return any_batch_partial;
}

/**
 * @brief Reset the outputs held in the workspace, releasing the current memory and setting the
 * effective batch size to 0 for every output.
 */
void ClearOutputs(Workspace &ws, const OpSpec &spec) {
  for (int i = 0; i < spec.NumOutput(); i++) {
    if (ws.template OutputIsType<CPUBackend>(i)) {
      ws.template Output<CPUBackend>(i).Reset();
    } else {
      ws.template Output<GPUBackend>(i).Reset();
    }
  }
}

}  // namespace

/**
 * @brief Takes the batch size from any of the op's tensor inputs.
 *
 * If no inputs were specified, a batch size inferred from
 * the stage queue is used instead.
 *
 * Assumes that most of the operators expect uniform batch
 * size between all inputs and outputs. The notable exception
 * of split and merge operators cannot rely on this value.
 */
inline int InferBatchSizeFromInput(const Workspace &ws, int stage_batch_size) {
  if (ws.NumInput() > 0) {
    return ws.GetInputBatchSize(0);
  }
  const ArgumentWorkspace &argument_ws = ws;
  if (HasTensorArgInputs(argument_ws)) {
    auto [name, arg] = *begin(argument_ws);
    return arg.tvec->num_samples();
  }
  return stage_batch_size;
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::PreRun() {
  auto batch_size = InferBatchSize(batch_size_providers_);
  batch_sizes_cpu_.push(batch_size);
  batch_sizes_mixed_.push(batch_size);
  batch_sizes_gpu_.push(batch_size);
}

template <typename WorkspacePolicy, typename QueuePolicy>
Executor<WorkspacePolicy, QueuePolicy>::~Executor() {
  Shutdown();
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::Shutdown() {
  try {
    SyncDevice();
  } catch (const CUDAError &e) {
    if (!e.is_unloading())
      throw;
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::SyncDevice() {
  if (device_id_ != CPU_ONLY_DEVICE_ID) {
    DeviceGuard dg(device_id_);
    if (mixed_op_stream_)
      CUDA_DTOR_CALL(cudaStreamSynchronize(mixed_op_stream_));
    if (gpu_op_stream_)
      CUDA_DTOR_CALL(cudaStreamSynchronize(gpu_op_stream_));
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunCPUImpl(size_t iteration_id) {
  PreRun();
  const char placement_error[] =
      "Cannot run a pipeline with Mixed/GPU ops in CPU-only mode. Please provide "
      "valid device id or change the operators' device.";
  if (device_id_ < 0) {
    DALI_ENFORCE(device_id_ == CPU_ONLY_DEVICE_ID,
                 "Wrong device_id provided, it should be >= 0, "
                 "or equal to CPU_ONLY_DEVICE_ID.");
    DALI_ENFORCE(graph_->NumOp(OpType::GPU) == 0, placement_error);

    for (int i = 0; i < graph_->NumOp(OpType::MIXED) && !exec_error_; ++i) {
      const OpNode &op_node = graph_->Node(OpType::MIXED, i);
      DALI_ENFORCE(op_node.spec.GetSchema().name() == "MakeContiguous", placement_error);
      for (auto tensor_id : op_node.children_tensors) {
        const TensorNode &tensor_node = graph_->Tensor(tensor_id);
        DALI_ENFORCE(tensor_node.producer.storage_device == StorageDevice::CPU, placement_error);
      }
    }
  }

  DomainTimeRange tr("[DALI][Executor] RunCPU");

  DeviceGuard g(device_id_);

  auto cpu_idxs = QueuePolicy::AcquireIdxs(OpType::CPU);
  if (exec_error_ || QueuePolicy::IsStopSignaled() ||
      !QueuePolicy::template AreValid<OpType::CPU>(cpu_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::CPU, cpu_idxs);
    return;
  }

  int stage_batch_size = batch_sizes_cpu_.front();
  batch_sizes_cpu_.pop();

  // Run the cpu-ops in the thread
  // Process each CPU Op in batch
  for (int cpu_op_id = 0; cpu_op_id < graph_->NumOp(OpType::CPU) && !exec_error_; ++cpu_op_id) {
    OpNode &op_node = graph_->Node(OpType::CPU, cpu_op_id);
    decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::CPU>(cpu_idxs, *graph_, cpu_op_id);

    int batch_size = InferBatchSizeFromInput(ws, stage_batch_size);
    ws.SetBatchSizes(batch_size);

    DomainTimeRange tr("[DALI][CPU op] " + op_node.instance_name, DomainTimeRange::kBlue1);

    try {
      RunHelper(op_node, ws, iteration_id);
      FillStats(cpu_memory_stats_, ws, "CPU_" + op_node.instance_name, cpu_memory_stats_mutex_);
    } catch (std::exception &e) {
      HandleError("CPU", op_node, e.what());
    } catch (...) {
      HandleError();
    }
  }

  // Pass the work to the mixed stage
  QueuePolicy::ReleaseIdxs(OpType::CPU, cpu_idxs);
}


template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunMixedImpl(size_t iteration_id) {
  DomainTimeRange tr("[DALI][Executor] RunMixed");
  DeviceGuard g(device_id_);

  auto mixed_idxs = QueuePolicy::AcquireIdxs(OpType::MIXED);
  if (exec_error_ || QueuePolicy::IsStopSignaled() ||
     !QueuePolicy::template AreValid<OpType::MIXED>(mixed_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::MIXED, mixed_idxs);
    return;
  }

  // Enforce our assumed dependency between consecutive
  // iterations of a stage of the pipeline.

  if (device_id_ != CPU_ONLY_DEVICE_ID)
    CUDA_CALL(cudaEventSynchronize(mixed_stage_event_));

  int stage_batch_size = batch_sizes_mixed_.front();
  batch_sizes_mixed_.pop();

  for (int i = 0; i < graph_->NumOp(OpType::MIXED) && !exec_error_; ++i) {
    OpNode &op_node = graph_->Node(OpType::MIXED, i);
    try {
      decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::MIXED>(mixed_idxs, *graph_, i);

      int batch_size = InferBatchSizeFromInput(ws, stage_batch_size);
      ws.SetBatchSizes(batch_size);

      DomainTimeRange tr("[DALI][Mixed op] " + op_node.instance_name, DomainTimeRange::kOrange);
      RunHelper(op_node, ws, iteration_id);
      FillStats(mixed_memory_stats_, ws, "MIXED_" + op_node.instance_name,
                mixed_memory_stats_mutex_);
      if (device_id_ != CPU_ONLY_DEVICE_ID) {
        if (ws.has_stream() && ws.has_event()) {
            CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
        }
        CUDA_CALL(cudaGetLastError());
      }
    } catch (std::exception &e) {
      HandleError("Mixed", op_node, e.what());
    } catch (...) {
      HandleError();
    }
  }

  if (device_id_ != CPU_ONLY_DEVICE_ID) {
    if (!mixed_output_events_.empty()) {
      int queue_id = mixed_idxs[OpType::MIXED];
      CUDA_CALL(cudaEventRecord(mixed_output_events_.GetEvent(queue_id), mixed_op_stream_));
    }

    // We know that this is the proper stream, we do not need to look it up in any workspace
    CUDA_CALL(cudaEventRecord(mixed_stage_event_, mixed_op_stream_));
  }

  // Pass the work to the gpu stage
  QueuePolicy::ReleaseIdxs(OpType::MIXED, mixed_idxs, mixed_op_stream_);
}


template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunGPUImpl(size_t iteration_id) {
  DomainTimeRange tr("[DALI][Executor] RunGPU");

  auto gpu_idxs = QueuePolicy::AcquireIdxs(OpType::GPU);
  if (exec_error_ || QueuePolicy::IsStopSignaled() ||
      !QueuePolicy::template AreValid<OpType::GPU>(gpu_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::GPU, gpu_idxs);
    return;
  }

  // short path for pure CPU pipeline
  if (device_id_ == CPU_ONLY_DEVICE_ID) {
    // We do not release, but handle to used outputs
    QueuePolicy::QueueOutputIdxs(gpu_idxs, gpu_op_stream_);
    return;
  }
  DeviceGuard g(device_id_);

  // Enforce our assumed dependency between consecutive
  // iterations of a stage of the pipeline.
  CUDA_CALL(cudaEventSynchronize(gpu_stage_event_));

  int stage_batch_size = batch_sizes_gpu_.front();
  batch_sizes_gpu_.pop();

  for (int i = 0; i < graph_->NumOp(OpType::GPU) && !exec_error_; ++i) {
    OpNode &op_node = graph_->Node(OpType::GPU, i);
    try {
      decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::GPU>(gpu_idxs, *graph_, i);

      int batch_size = InferBatchSizeFromInput(ws, stage_batch_size);
      ws.SetBatchSizes(batch_size);

      auto parent_events = ws.ParentEvents();

      for (auto &event : parent_events) {
        CUDA_CALL(cudaStreamWaitEvent(ws.stream(), event, 0));
      }

      DomainTimeRange tr("[DALI][GPU op] " + op_node.instance_name, DomainTimeRange::knvGreen);
      RunHelper(op_node, ws, iteration_id);
      FillStats(gpu_memory_stats_, ws, "GPU_" + op_node.instance_name, gpu_memory_stats_mutex_);
      if (ws.has_event()) {
        CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
      }
      CUDA_CALL(cudaGetLastError());
    } catch (std::exception &e) {
      HandleError("GPU", op_node, e.what());
    } catch (...) {
      HandleError();
    }
  }

  // Update the ready queue to signal that all the work
  // in the `gpu_idxs` set of output buffers has been
  // issued. Notify any waiting threads.

  // If we have GPU outputs than
  if (!gpu_output_events_.empty()) {
    int queue_id = gpu_idxs[OpType::GPU];
    CUDA_CALL(cudaEventRecord(gpu_output_events_.GetEvent(queue_id), gpu_op_stream_));
  }

  // We know that this is the proper stream, we do not need to look it up in any workspace
  CUDA_CALL(cudaEventRecord(gpu_stage_event_, gpu_op_stream_));

  // We do not release, but handle to used outputs
  QueuePolicy::QueueOutputIdxs(gpu_idxs, gpu_op_stream_);
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunCPU() {
  try {
    RunCPUImpl(cpu_iteration_id_++);
  } catch (std::exception &e) {
    HandleError(make_string("Exception in CPU stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in CPU stage.");
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunMixed() {
  try {
    RunMixedImpl(mixed_iteration_id_++);
  } catch (std::exception &e) {
    HandleError(make_string("Exception in mixed stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in mixed stage.");
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunGPU() {
  try {
    RunGPUImpl(gpu_iteration_id_++);
  } catch (std::exception &e) {
    HandleError(make_string("Exception in GPU stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in GPU stage.");
  }
}


template<typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunHelper(OpNode &op_node, Workspace &ws,
                                                       size_t iteration_id) {
  auto &output_desc = op_node.output_desc;
  auto &op = *op_node.op;
  output_desc.clear();
  const auto &spec = op.GetSpec();
  const auto &schema = spec.GetSchema();
  SmallVector<int, 16> empty_layout_in_idxs;

  // Create a checkpoint for the given iteration.
  auto create_checkpoint = [&](int iter) {
    auto &cpt = GetCurrentIterationData(iter).checkpoint;
    auto &op_cpt = cpt.GetOpCheckpoint(op_node.id);
    cpt.SetIterationId(iter);
    op_node.op->SaveState(op_cpt, ws.has_stream() ? std::optional{ws.stream()} : std::nullopt);
  };

  // If it is the first iteration, create initial checkpoint.
  // This way, we make sure there is always a checkpoint that can be accessed.
  if (checkpointing_ && iteration_id == 0)
    create_checkpoint(iteration_id);

  ws.InjectOperatorTraces(GetCurrentIterationData(iteration_id).operator_traces);
  ws.ClearOperatorTraces();

  auto ws_order = ws.has_stream() ? AccessOrder(ws.stream()) : AccessOrder::host();

  // Try to infer from which stage the buffer comes from
  auto order_to_stage_index = [&](AccessOrder order) {
    if (!order || order == AccessOrder::host())
      return 0;
    else if (order == mixed_op_stream_)
      return 1;
    else if (order == gpu_op_stream_)
      return 2;
    else
      return 3;  // foreign order, comes last
  };

  auto set_order = [&](auto &output, AccessOrder order) {
    // Check if this stage (identified by ws_order) is already synchronized with the output's
    // order. If yes, we're implicitly synchronized and we can skip sync.
    bool need_sync = order_to_stage_index(output.order()) > order_to_stage_index(ws_order);
    output.set_order(order, need_sync);
  };

  for (int i = 0; i < ws.NumOutput(); i++) {
    if (ws.OutputIsType<CPUBackend>(i)) {
      set_order(ws.Output<CPUBackend>(i), AccessOrder::host());
    } else {
      set_order(ws.Output<GPUBackend>(i), ws_order);
    }
  }

  // Assuming that most operators don't expect empty input, and expect consistent input.
  if (ws.NumInput() > 0 || HasTensorArgInputs(ws)) {
    if (AllInputsEmpty(ws)) {
      // We skip the execution of this operator and Reset the outputs in case some state was still
      // present.
      ClearOutputs(ws, spec);
      // TODO(klecki): Instead of skipping the execution, rework all DALI operators to correctly
      // propagate the dim, type and do validation (arguments, types, etc) with empty input batches.
      return;
    }
  }

  for (int i = 0; i < ws.NumInput(); i++) {
    DALI_ENFORCE(
        ws.GetInputBatchSize(i) <= max_batch_size_,
        make_string("Expected batch size lower or equal to max batch size. Expected at most: ",
                    max_batch_size_, ", got: ", ws.GetInputBatchSize(i)));
  }
  for (int i = 0; i < spec.NumOutput(); i++) {
    DALI_ENFORCE(ws.GetRequestedBatchSize(i) <= max_batch_size_,
                 make_string("Expected batch size lower or equal to max batch size. Actual: ",
                             ws.GetRequestedBatchSize(i), " <= ", max_batch_size_));
  }

  // If we are using conditional statements try to keep smaller memory footprint by not keeping
  // the largest batch produced in conditional block as reserved memory.
  // If the conditionals were detected in this graph (split operator with `_if_stmt` argument)
  // see if the operator is executed in conditional scope - seeing a partial batch processed
  // would indicate it in most cases (unless someone mixes partial batches and conditionals).
  // In such case reset the outputs before adjusting their size so we free the reserved memory
  // and allocated only batch of currently needed size.
  if (HasConditionals() && AnyBatchPartial(ws, spec, max_batch_size_)) {
    ClearOutputs(ws, spec);
  }

  for (int i = 0; i < spec.NumRegularInput(); i++) {
    bool had_empty_layout = false;
    if (ws.InputIsType<CPUBackend>(i)) {
      had_empty_layout =
          SetDefaultLayoutIfNeeded(ws.UnsafeMutableInput<CPUBackend>(i), schema, i);
    } else {
      had_empty_layout =
          SetDefaultLayoutIfNeeded(ws.UnsafeMutableInput<GPUBackend>(i), schema, i);
    }
    if (had_empty_layout) empty_layout_in_idxs.push_back(i);
  }

  bool should_allocate = false;
  {
    DomainTimeRange tr("[DALI][Executor] Setup");
    should_allocate = op.Setup(output_desc, ws);
  }
  {
    DomainTimeRange tr("[DALI][Executor] Allocate outputs");
    if (should_allocate) {
      DALI_ENFORCE(
          static_cast<size_t>(ws.NumOutput()) == output_desc.size(),
          "Operator::Setup returned shape and type information for mismatched number of outputs");
      DALI_ENFORCE(op.CanInferOutputs(),
                    "Operator::Setup returned true indicating that it successfully calculated "
                    "shape and type information for Operator outputs. In that case "
                    "CanInferOutputs should always return true.");
      for (int i = 0; i < ws.NumOutput(); i++) {
        auto &desc = output_desc[i];
        if (ws.OutputIsType<CPUBackend>(i)) {
          ws.Output<CPUBackend>(i).Resize(desc.shape, desc.type);
        } else {
          ws.Output<GPUBackend>(i).Resize(desc.shape, desc.type);
        }
      }
    } else {
      DALI_ENFORCE(!op.CanInferOutputs(),
                    "Operator::Setup returned false indicating that it cannot calculate shape and "
                    "type information for Operator outputs. In that case CanInferOutputs should "
                    "always return false.");
    }
  }

  ClearOutputSourceInfo(ws);

  {
    DomainTimeRange tr("[DALI][Executor] Run");
    op.Run(ws);
  }

  PropagateSourceInfo(ws);

  /* TODO(michalz): Find a way to make this valid in presence of passthrough between stages
  // Set the output order to the stage's stream
  for (int i = 0; i < ws.NumOutput(); i++) {
    if (ws.OutputIsType<CPUBackend>(i)) {
      ws.Output<CPUBackend>(i).set_order(ws_order);
    } else {
      ws.Output<GPUBackend>(i).set_order(ws_order);
    }
  }
  */

  for (int i : empty_layout_in_idxs) {
    if (ws.InputIsType<CPUBackend>(i)) {
      auto &in = ws.UnsafeMutableInput<CPUBackend>(i);
      in.SetLayout({});
    } else {
      auto &in = ws.UnsafeMutableInput<GPUBackend>(i);
      in.SetLayout({});
    }
  }

  // If it is the end of an epoch, create a checkpoint.
  // The checkpoint corresponds to the state between the iteration `iteration_id`
  // and `iteration_id + 1`.
  // After consuming all the outputs from the epoch, GetCurrentCheckpoint is going to
  // return the created checkpoint.
  if (checkpointing_ && (iteration_id + 1) % checkpointing_epoch_size_ == 0)
    create_checkpoint(iteration_id + 1);
}


template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::ShareOutputsImpl(Workspace *ws, size_t iteration_id) {
  DALI_ENFORCE(ws != nullptr, "Workspace is nullptr");
  DeviceGuard g(device_id_);
  ws->Clear();

  if (exec_error_ || QueuePolicy::IsStopSignaled())
    RethrowError();

  auto output_idx = QueuePolicy::UseOutputIdxs();

  if (exec_error_ || QueuePolicy::IsStopSignaled())
    RethrowError();

  // We need to fill the output workspace with pointers to appropriate output buffers.
  for (size_t i = 0; i < pipeline_outputs_.size(); i++) {
    auto out_tensor_id = pipeline_outputs_[i];
    auto &out_tensor = graph_->Tensor(out_tensor_id);
    auto op_type = graph_->Node(out_tensor.producer.node).op_type;
    auto storage_dev = out_tensor.producer.storage_device;
    VALUE_SWITCH(storage_dev, storage_dev_static, (StorageDevice::GPU, StorageDevice::CPU), (
      VALUE_SWITCH(op_type, op_type_static, (OpType::CPU, OpType::MIXED, OpType::GPU), (
        auto &queue =
               get_queue<op_type_static, storage_dev_static>(tensor_to_store_queue_[out_tensor_id]);
        auto stage_output_idx = output_idx[op_type_static];
        ws->AddOutput(queue[stage_output_idx]);
      ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
    ), DALI_FAIL("Invalid storage device"));  // NOLINT(whitespace/parens)
  }

  ws->InjectOperatorTraces(GetCurrentIterationData(iteration_id).operator_traces);


  // Mostly a sanity check - we don't want to return a non-contiguous batch to Python.
  for (int i = 0; i < ws->NumOutput(); i++) {
    const char *error_msg =
            "DALI internal error: all outputs from the Pipeline must be contiguous after being "
            "processed by MakeContiguous operator.";
    if (ws->OutputIsType<CPUBackend>(i)) {
      DALI_ENFORCE(ws->Output<CPUBackend>(i).IsContiguous(), error_msg);
    } else {
      DALI_ENFORCE(ws->Output<GPUBackend>(i).IsContiguous(), error_msg);
    }
  }

  // We than need to wait for GPU outputs from Mixed & GPU stages that are computed asynchronously.
  // If the output event list is not empty, it means that there are outputs on GPU that we
  // have to wait for.
  AccessOrder sync_order = ws->has_stream() ? AccessOrder(ws->stream()) : AccessOrder::host();

  if (!mixed_output_events_.empty()) {
    auto queue_idx = output_idx[OpType::MIXED];
    sync_order.wait(mixed_output_events_.GetEvent(queue_idx));
  }
  if (!gpu_output_events_.empty()) {
    auto queue_idx = output_idx[OpType::GPU];
    sync_order.wait(gpu_output_events_.GetEvent(queue_idx));
  }
}


template <typename WorkspacePolicy, typename QueuePolicy>
int Executor<WorkspacePolicy, QueuePolicy>::InferBatchSize(
    const std::vector<BatchSizeProvider *> &bsps) const {
  if (bsps.empty()) {
    return max_batch_size_;
  }
  for (size_t i = 1; i < bsps.size(); i++) {
    DALI_ENFORCE(bsps[0]->NextBatchSize() == bsps[i]->NextBatchSize(),
                 "Batch size must be uniform across an iteration");
  }
  int batch_size;
  try {
    batch_size = bsps[0]->NextBatchSize();
    for (auto &bsp : bsps) {
      bsp->Advance();
    }
  } catch (const std::out_of_range &e) {
    DALI_FAIL(
        make_string("Failed to acquire the next batch. Make sure, that DALI Pipeline is fed "
                    "with sufficient amount of data. ", e.what()));
  }
  return batch_size;
}


template<typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::InitIterationData() {
  cpu_iteration_id_ = 0;
  mixed_iteration_id_ = 0;
  gpu_iteration_id_ = 0;
  output_iteration_id_ = 0;
  size_t iteration_data_size = CalcIterationDataSize();
  iteration_data_.resize(iteration_data_size);
  for (auto& id : iteration_data_) {
    id.operator_traces = std::make_shared<operator_trace_map_t>();
    if (checkpointing_)
      id.checkpoint.Build(*graph_);
  }
}


template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::DetectConditionals() {
  for (Index i = 0; i < graph_->NumOp(); i++) {
    const auto &spec = graph_->Node(i).spec;
    if (IsSplit(spec.GetSchema())) {
      if (spec.GetArgument<bool>("_if_stmt")) {
        has_conditionals_ = true;
      }
    }
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
bool Executor<WorkspacePolicy, QueuePolicy>::HasConditionals() const {
  return has_conditionals_;
}

template<typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::InitCheckpointing() {
  if (!checkpointing_)
    return;

  if (std::is_same_v<QueuePolicy, SeparateQueuePolicy>)
    DALI_FAIL("Checkpointing is not supported with `separated` pipeline exection mode enabled. ")

  checkpointing_epoch_size_ = -1;
  std::string reader_name;
  for (const auto &node : graph_->GetOpNodes()) {
    auto meta = node.op->GetReaderMeta();
    if (meta.epoch_size_padded <= 0)
      continue;

    int local_epoch_size = (meta.epoch_size_padded + max_batch_size_ - 1) / max_batch_size_;
    if (checkpointing_epoch_size_ == -1) {
      checkpointing_epoch_size_ = local_epoch_size;
      reader_name = node.spec.name();
    } else if (checkpointing_epoch_size_ != local_epoch_size) {
      DALI_FAIL(make_string(
        "When the checkpointing is enabled, all readers must have the same epoch size. ",
        "The readers ", reader_name, " and ", node.spec.name(), " have different epoch sizes ",
        "(", checkpointing_epoch_size_, " and ", local_epoch_size, " respectively). "));
    }
  }

  /* If there is no operator with ReaderMeta, set the epoch size to 1. */
  if (checkpointing_epoch_size_ == -1)
    checkpointing_epoch_size_ = 1;
}

template<typename WorkspacePolicy, typename QueuePolicy>
Checkpoint &Executor<WorkspacePolicy, QueuePolicy>::GetCurrentCheckpoint() {
  auto &cpt = GetCurrentIterationData(output_iteration_id_).checkpoint;
  // Sanity check
  DALI_ENFORCE(cpt.GetIterationId() == output_iteration_id_,
               "Requested checkpoint does not correspond to the matching iteration. ");
  return cpt;
}

template<typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RestoreStateFromCheckpoint(const Checkpoint &cpt) {
  DALI_ENFORCE(cpu_iteration_id_ == 0, "Cannot restore state of a running executor. ");
  for (int i = 0; i < graph_->NumOp(); ++i)
    graph_->Node(i).op->RestoreState(cpt.GetOpCheckpoint(i));
}

template<typename WorkspacePolicy, typename QueuePolicy>
IterationData &
Executor<WorkspacePolicy, QueuePolicy>::GetCurrentIterationData(size_t iteration_id) {
  return iteration_data_[iteration_id % iteration_data_.size()];
}


template<typename WorkspacePolicy, typename QueuePolicy>
size_t Executor<WorkspacePolicy, QueuePolicy>::CalcIterationDataSize() const {
  /*
   * In SimpleExecutor it is possible to set queue depth higher than one. This does not impact
   * the concurrency of execution of stages, however it impacts how user can read the output data.
   * Essentially, when queue depth is higher than one, user can still call `daliRun` multiple times
   * and expect that the output will be there.
   *
   * Therefore, the IterationDataSize shall be `cpu_size + 1`.
   * The `+1` is for the output Workspace.
   */
  return this->queue_sizes_.cpu_size + 1;
}


template class DLL_PUBLIC Executor<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;
template class DLL_PUBLIC Executor<AOT_WS_Policy<SeparateQueuePolicy>, SeparateQueuePolicy>;

}  // namespace dali
