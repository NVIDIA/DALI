// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

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
void Executor<WorkspacePolicy, QueuePolicy>::RunCPUImpl() {
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

  auto batch_size = batch_sizes_cpu_.front();
  batch_sizes_cpu_.pop();

  // Run the cpu-ops in the thread
  // Process each CPU Op in batch
  for (int cpu_op_id = 0; cpu_op_id < graph_->NumOp(OpType::CPU) && !exec_error_; ++cpu_op_id) {
    OpNode &op_node = graph_->Node(OpType::CPU, cpu_op_id);
    decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::CPU>(cpu_idxs, *graph_, cpu_op_id);

    ws.SetBatchSizes(batch_size);

    DomainTimeRange tr("[DALI][CPU op] " + op_node.instance_name, DomainTimeRange::kBlue1);

    try {
      RunHelper(op_node, ws);
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
void Executor<WorkspacePolicy, QueuePolicy>::RunMixedImpl() {
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

  auto batch_size = batch_sizes_mixed_.front();
  batch_sizes_mixed_.pop();

  for (int i = 0; i < graph_->NumOp(OpType::MIXED) && !exec_error_; ++i) {
    OpNode &op_node = graph_->Node(OpType::MIXED, i);
    try {
      decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::MIXED>(mixed_idxs, *graph_, i);

      ws.SetBatchSizes(batch_size);

      DomainTimeRange tr("[DALI][Mixed op] " + op_node.instance_name, DomainTimeRange::kOrange);
      RunHelper(op_node, ws);
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
void Executor<WorkspacePolicy, QueuePolicy>::RunGPUImpl() {
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

  auto batch_size = batch_sizes_gpu_.front();
  batch_sizes_gpu_.pop();

  for (int i = 0; i < graph_->NumOp(OpType::GPU) && !exec_error_; ++i) {
    OpNode &op_node = graph_->Node(OpType::GPU, i);
    try {
      decltype(auto) ws = ws_policy_.template GetWorkspace<OpType::GPU>(gpu_idxs, *graph_, i);

      ws.SetBatchSizes(batch_size);

      auto parent_events = ws.ParentEvents();

      for (auto &event : parent_events) {
        CUDA_CALL(cudaStreamWaitEvent(ws.stream(), event, 0));
      }

      DomainTimeRange tr("[DALI][GPU op] " + op_node.instance_name, DomainTimeRange::knvGreen);
      RunHelper(op_node, ws);
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
    RunCPUImpl();
  } catch (std::exception &e) {
    HandleError(make_string("Exception in CPU stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in CPU stage.");
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunMixed() {
  try {
    RunMixedImpl();
  } catch (std::exception &e) {
    HandleError(make_string("Exception in mixed stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in mixed stage.");
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunGPU() {
  try {
    RunGPUImpl();
  } catch (std::exception &e) {
    HandleError(make_string("Exception in GPU stage: ", e.what()));
  } catch (...) {
    HandleError("Unknown error in GPU stage.");
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunHelper(OpNode &op_node, Workspace &ws) {
  auto &output_desc = op_node.output_desc;
  auto &op = *op_node.op;
  output_desc.clear();
  const auto &spec = op.GetSpec();
  const auto &schema = spec.GetSchema();
  SmallVector<int, 16> empty_layout_in_idxs;

  cudaStream_t prev_stage_stream = ws.has_stream() && ws.stream() == gpu_op_stream_
    ? mixed_op_stream_ : gpu_op_stream_;

  auto order = ws.has_stream() ? AccessOrder(ws.stream()) : AccessOrder::host();
  auto set_order = [&](auto &output) {
    // NOTE: the stage streams are synchronized by the executor
    bool need_sync = output.order().stream() != prev_stage_stream;
    output.set_order(order, need_sync);
  };

  for (int i = 0; i < ws.NumOutput(); i++) {
    if (ws.OutputIsType<CPUBackend>(i)) {
      set_order(ws.Output<CPUBackend>(i));
    } else {
      set_order(ws.Output<GPUBackend>(i));
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
  {
    DomainTimeRange tr("[DALI][Executor] Run");
    op.Run(ws);
  }

  for (int i : empty_layout_in_idxs) {
    if (ws.InputIsType<CPUBackend>(i)) {
      auto &in = ws.UnsafeMutableInput<CPUBackend>(i);
      in.SetLayout({});
    } else {
      auto &in = ws.UnsafeMutableInput<GPUBackend>(i);
      in.SetLayout({});
    }
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
    assert(batch_size > 0);
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

template class DLL_PUBLIC Executor<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;
template class DLL_PUBLIC Executor<AOT_WS_Policy<SeparateQueuePolicy>, SeparateQueuePolicy>;

}  // namespace dali
