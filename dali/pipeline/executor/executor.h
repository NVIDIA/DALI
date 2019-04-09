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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_H_

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/executor/queue_policy.h"
#include "dali/pipeline/executor/workspace_policy.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/util/stream_pool.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {


namespace detail {
// This is stream callback used on GPU stream to indicate that GPU work for this
// pipeline run is finished
static void gpu_finished_callback(cudaStream_t stream, cudaError_t status, void *userData);

}  // namespace detail

class DLL_PUBLIC ExecutorBase {
 public:
  using ExecutorCallback = std::function<void(void)>;
  DLL_PUBLIC virtual ~ExecutorBase() noexcept(false) {}
  DLL_PUBLIC virtual void Build(OpGraph *graph, vector<string> output_names) = 0;
  DLL_PUBLIC virtual void Init() = 0;
  DLL_PUBLIC virtual void RunCPU() = 0;
  DLL_PUBLIC virtual void RunMixed() = 0;
  DLL_PUBLIC virtual void RunGPU() = 0;
  DLL_PUBLIC virtual void Outputs(DeviceWorkspace *ws) = 0;
  DLL_PUBLIC virtual void ShareOutputs(DeviceWorkspace *ws) = 0;
  DLL_PUBLIC virtual void ReleaseOutputs() = 0;
  DLL_PUBLIC virtual void SetCompletionCallback(ExecutorCallback cb) = 0;

 protected:
  // virtual to allow the TestPruneWholeGraph test in gcc
  virtual void PruneUnusedGraphNodes() = 0;

  template <typename T>
  friend class ExecutorTest;
};

/**
 * @brief Basic executor for dali graphs. This executor enables
 * prefetching of results by maintaining two copies of output
 * buffers, so that we can produce data into one while the
 * other is in use by the user.
 */
template <typename WorkspacePolicy, typename QueuePolicy>
class DLL_PUBLIC Executor : public ExecutorBase, public WorkspacePolicy, public QueuePolicy {
 public:
  DLL_PUBLIC inline Executor(int batch_size, int num_thread, int device_id,
                             size_t bytes_per_sample_hint, bool set_affinity = false,
                             int max_num_stream = -1, int default_cuda_stream_priority = 0,
                             QueueSizes prefetch_queue_depth = QueueSizes{2, 2})
      : batch_size_(batch_size),
        device_id_(device_id),
        bytes_per_sample_hint_(bytes_per_sample_hint),
        callback_(nullptr),
        stream_pool_(max_num_stream, true, default_cuda_stream_priority),
        event_pool_(max_num_stream),
        thread_pool_(num_thread, device_id, set_affinity),
        exec_error_(false),
        queue_sizes_(prefetch_queue_depth) {
    DALI_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    DALI_ENFORCE(device_id >= 0, "Device id must be non-negative.");

    stage_queue_depths_ = QueuePolicy::GetQueueSizes(prefetch_queue_depth);
  }

  DLL_PUBLIC void Build(OpGraph *graph, vector<string> output_names) override;
  DLL_PUBLIC void Init() override {}
  DLL_PUBLIC void RunCPU() override;
  DLL_PUBLIC void RunMixed() override;
  DLL_PUBLIC void RunGPU() override;
  DLL_PUBLIC void Outputs(DeviceWorkspace *ws) override;
  DLL_PUBLIC void ShareOutputs(DeviceWorkspace *ws) override;
  DLL_PUBLIC void ReleaseOutputs() override;
  DLL_PUBLIC void SetCompletionCallback(ExecutorCallback cb) override;

  DLL_PUBLIC void ShutdownQueue() {
    QueuePolicy::SignalStop();
  }

  DISABLE_COPY_MOVE_ASSIGN(Executor);

 protected:
  void PruneUnusedGraphNodes() override;

  virtual std::vector<int> GetTensorQueueSizes(const OpGraph &graph);

  virtual void SetupOutputInfo(const OpGraph &graph);

  std::vector<int> GetMemoryHints(const OpNode &node);

  void PrepinData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                  const OpGraph &graph);

  void PresizeData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                   const OpGraph &graph);

  void SetupOutputQueuesForGraph();

  class EventList {
   public:
    inline EventList() {}
    inline EventList(int size, EventPool *event_pool) {
      DALI_ENFORCE(event_pool != nullptr);
      for (int i = 0; i < size; ++i) {
        events_.push_back(event_pool->GetEvent());
      }
    }

    inline cudaEvent_t GetEvent(int idx) { return events_[idx]; }

   private:
    vector<cudaEvent_t> events_;
  };

  int batch_size_, device_id_;
  size_t bytes_per_sample_hint_;
  int previous_gpu_queue_idx_ = -1;

  vector<string> output_names_;

  // Meta-data about our stage outputs for fast lookup
  std::vector<TensorNodeId> pipeline_outputs_;
  std::vector<EventList> gpu_output_events_;

  // Work is passed between the stages through queues. This
  // is needed for potentially asynchronous work issue, which
  // some Executors that derive from this class implement.
  //
  // In the case that work issue is pipelined, a stages issue
  // could run at the same time as the next iterations issue
  // for the previous stage. To avoid thread-safety issues
  // with updating our queues, we need to lock when we update
  // them. However, this executor assumes the same thread
  // will call Run*, so it does not block if no work exists
  // for the stage that was called (it will throw an error).
  //
  // Derived executors that implement asynchronous work issue
  // must handle their own synchronization between the same
  // iteration of each stage. While it is not ideal to have
  // two sets of locks doing similar things in each stage,
  // it simplifies the software for now so we leave it
  // unless it becomes an issue in the future.


  StageQueues stage_queue_depths_;

  OpGraph *graph_ = nullptr;
  // we need to keep this above the stream_pool_ so we still have it when the stream_pool_
  // destructor runs and it waits for streams to finish
  ExecutorCallback callback_;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
  std::vector<std::string> errors_;
  std::mutex errors_mutex_;
  bool exec_error_;
  QueueSizes queue_sizes_;
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  // MixedOpId -> queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED to GPU Ops
  std::vector<std::vector<cudaEvent_t>> mixed_op_events_;
  // queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED stage to GPU stage for callback only
  // in some edge cases where there are no operators
  std::vector<cudaEvent_t> mixed_callback_events_;
};

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::SetCompletionCallback(ExecutorCallback cb) {
  callback_ = cb;
  // Create necessary events lazily
  if (mixed_callback_events_.empty()) {
    mixed_callback_events_.resize(stage_queue_depths_[OpType::MIXED]);
    for (auto &event : mixed_callback_events_) {
      event = event_pool_.GetEvent();
    }
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::Build(OpGraph *graph, vector<string> output_names) {
  DALI_ENFORCE(graph != nullptr, "Input graph is nullptr.");
  DALI_ENFORCE(graph->NumOp() > 0, "Graph has no operators.");
  graph->InstantiateOperators();  // ..if not done already

  output_names_ = output_names;
  graph_ = graph;

  DeviceGuard g(device_id_);

  // Remove any node from the graph whose output
  // will not be used as an output or by another node
  PruneUnusedGraphNodes();

  // Check if graph is ok for execution
  CheckGraphConstraints(*graph_);
  // Clear the old data
  tensor_to_store_queue_.clear();

  // TODO(klecki) this setups the event queues as well
  SetupOutputInfo(*graph_);

  auto queue_sizes = GetTensorQueueSizes(*graph_);

  // Create corresponding storage type for TensorNodes in graph
  tensor_to_store_queue_ = CreateBackingStorageForTensorNodes(*graph_, batch_size_, queue_sizes);
  // Setup stream and events that will be used for execution
  {
    DeviceGuard g(device_id_);
    mixed_op_stream_ = stream_pool_.GetStream();
    gpu_op_stream_ = stream_pool_.GetStream();
    mixed_op_events_ =
        CreateEventsForMixedOps(event_pool_, *graph_, stage_queue_depths_[OpType::MIXED]);
  }

  PrepinData(tensor_to_store_queue_, *graph_);

  // Presize the workspaces based on the hint
  PresizeData(tensor_to_store_queue_, *graph_);

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  // For each set of outputs, setup another set of
  // workspaces so that nothing has to be altered
  // during execution (this is necessary for
  // asynchonrous executors that can overlap work issue)
  WorkspacePolicy::InitializeWorkspaceStore(*graph_, tensor_to_store_queue_, mixed_op_stream_,
                                            gpu_op_stream_, mixed_op_events_, queue_sizes_);

  // Producer-consumer queues info
  SetupOutputQueuesForGraph();
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunCPU() {
  TimeRange tr("[Executor] RunCPU");

  auto support_idxs = QueuePolicy::AcquireIdxs(OpType::SUPPORT);
  if (exec_error_ || QueuePolicy::IsStopSignaled() || !QueuePolicy::AreValid(support_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::SUPPORT, support_idxs);
    return;
  }

  DeviceGuard g(device_id_);

  // Run the support ops
  try {
    for (int i = 0; i < graph_->NumOp(OpType::SUPPORT); ++i) {
      OpNode &op_node = graph_->Node(OpType::SUPPORT, i);
      OperatorBase &op = *op_node.op;
      // SupportWorkspace &ws = GetWorkspace<OpType::SUPPORT>(queue_idx, i);
      typename WorkspacePolicy::template ws_t<OpType::SUPPORT> ws =
          WorkspacePolicy::template GetWorkspace<OpType::SUPPORT>(support_idxs, *graph_, i);
      TimeRange tr("[Executor] Run Support op " + op_node.instance_name,
          TimeRange::kCyan);
      op.Run(&ws);
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    QueuePolicy::SignalStop();
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    // Let the ReleaseIdx chain wake the output cv
  }

  QueuePolicy::ReleaseIdxs(OpType::SUPPORT, support_idxs);

  auto cpu_idxs = QueuePolicy::AcquireIdxs(OpType::CPU);
  if (exec_error_ || QueuePolicy::IsStopSignaled() || !QueuePolicy::AreValid(cpu_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::CPU, cpu_idxs);
    return;
  }

  // Run the cpu-ops in the thread pool
  for (int i = 0; i < batch_size_; ++i) {
    thread_pool_.DoWorkWithID(std::bind(
          [this, cpu_idxs] (int data_idx, int tid) {
          TimeRange tr("[Executor] RunCPU on " + to_string(data_idx));
          SampleWorkspace ws;
          for (int j = 0; j < graph_->NumOp(OpType::CPU); ++j) {
            OpNode &op_node = graph_->Node(OpType::CPU, j);
            OperatorBase &op = *op_node.op;
            WorkspacePolicy::template GetWorkspace<OpType::CPU>(cpu_idxs, *graph_, op_node)
                .GetSample(&ws, data_idx, tid);
            TimeRange tr("[Executor] Run CPU op " + op_node.instance_name
                + " on " + to_string(data_idx),
                TimeRange::kBlue1);
            op.Run(&ws);
          }
          }, i, std::placeholders::_1));
  }
  try {
    thread_pool_.WaitForWork();
  } catch (std::runtime_error& e) {
    exec_error_ = true;
    QueuePolicy::SignalStop();
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    // Let the ReleaseIdx chain wake the output cv
  }
  // Pass the work to the mixed stage
  QueuePolicy::ReleaseIdxs(OpType::CPU, cpu_idxs);
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunMixed() {
  TimeRange tr("[Executor] RunMixed");
  DeviceGuard g(device_id_);

  auto mixed_idxs = QueuePolicy::AcquireIdxs(OpType::MIXED);
  if (exec_error_ || QueuePolicy::IsStopSignaled() || !QueuePolicy::AreValid(mixed_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::MIXED, mixed_idxs);
    return;
  }

  try {
    for (int i = 0; i < graph_->NumOp(OpType::MIXED); ++i) {
      OpNode &op_node = graph_->Node(OpType::MIXED, i);
      OperatorBase &op = *op_node.op;
      typename WorkspacePolicy::template ws_t<OpType::MIXED> ws =
          WorkspacePolicy::template GetWorkspace<OpType::MIXED>(mixed_idxs, *graph_, i);
      TimeRange tr("[Executor] Run Mixed op " + op_node.instance_name,
          TimeRange::kOrange);
      op.Run(&ws);
      if (ws.has_stream() && ws.has_event()) {
        CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
      }
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    QueuePolicy::SignalStop();
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    // Let the ReleaseIdx chain wake the output cv
  }

  if (callback_) {
    // Record event that will allow to call the callback after whole run of this pipeline is
    // finished.
    CUDA_CALL(cudaEventRecord(mixed_callback_events_[mixed_idxs[OpType::MIXED]], mixed_op_stream_));
  }
  // Pass the work to the gpu stage
  QueuePolicy::ReleaseIdxs(OpType::MIXED, mixed_idxs, mixed_op_stream_);
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::RunGPU() {
  TimeRange tr("[Executor] RunGPU");

  auto gpu_idxs = QueuePolicy::AcquireIdxs(OpType::GPU);
  if (exec_error_ || QueuePolicy::IsStopSignaled() || !QueuePolicy::AreValid(gpu_idxs)) {
    QueuePolicy::ReleaseIdxs(OpType::GPU, gpu_idxs);
    return;
  }
  DeviceGuard g(device_id_);

  // Enforce our assumed dependency between consecutive
  // iterations of a stage of the pipeline.
  if (previous_gpu_queue_idx_ != -1) {
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (graph_->TensorIsType<CPUBackend>(output_names_[i])) continue;
      CUDA_CALL(cudaEventSynchronize(
              gpu_output_events_[i].GetEvent(previous_gpu_queue_idx_)));
    }
  }

  try {
    for (int i = 0; i < graph_->NumOp(OpType::GPU); ++i) {
      OpNode &op_node = graph_->Node(OpType::GPU, i);
      OperatorBase &op = *op_node.op;
      typename WorkspacePolicy::template ws_t<OpType::GPU> ws =
          WorkspacePolicy::template GetWorkspace<OpType::GPU>(gpu_idxs, *graph_, i);
      auto parent_events = ws.ParentEvents();

      for (auto &event : parent_events) {
        CUDA_CALL(cudaStreamWaitEvent(ws.stream(), event, 0));
      }

      TimeRange tr("[Executor] Run GPU op " + op_node.instance_name,
          TimeRange::knvGreen);
      op.Run(&ws);
      if (ws.has_event()) {
        CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
      }
    }

    // TODO(klecki): do not go over string names, please
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (graph_->TensorIsType<CPUBackend>(output_names_[i])) continue;
      OpNodeId src_id = graph_->TensorSourceID(output_names_[i]);
      int src_idx = graph_->NodeIdx(src_id);

      // Record events for each output requested by the user
      cudaEvent_t event = gpu_output_events_[i].GetEvent(gpu_idxs[OpType::GPU]);
      if (graph_->NodeType(src_id) == OpType::MIXED) {
        typename WorkspacePolicy::template ws_t<OpType::MIXED> ws =
            WorkspacePolicy::template GetWorkspace<OpType::MIXED>(gpu_idxs, *graph_, src_idx);
        CUDA_CALL(cudaEventRecord(event, ws.stream()));
      } else if (graph_->NodeType(src_id) == OpType::GPU) {
        typename WorkspacePolicy::template ws_t<OpType::GPU> ws =
            WorkspacePolicy::template GetWorkspace<OpType::GPU>(gpu_idxs, *graph_, src_idx);
        CUDA_CALL(cudaEventRecord(event, ws.stream()));
      } else {
        DALI_FAIL("Internal error. Output node is not gpu/mixed");
      }
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    QueuePolicy::SignalStop();
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    // Let the ReleaseIdx chain wake the output cv
  }
  // Update the ready queue to signal that all the work
  // in the `gpu_idxs` set of output buffers has been
  // issued. Notify any waiting threads.

  // Schedule the call to any callback registered previously
  if (callback_) {
    CUDA_CALL(cudaStreamWaitEvent(gpu_op_stream_,
                                  mixed_callback_events_[gpu_idxs[OpType::MIXED]], 0));
    CUDA_CALL(cudaStreamAddCallback(gpu_op_stream_, &detail::gpu_finished_callback,
                                    static_cast<void *>(&callback_), 0));
  }

  // We do not release, but handle to used outputs
  QueuePolicy::QueueOutputIdxs(gpu_idxs, gpu_op_stream_);

  // Save the gpu_idxs so we can enforce the
  // dependency between consecutive iterations
  // of the gpu stage of the pipeline.
  previous_gpu_queue_idx_ = gpu_idxs[OpType::GPU];
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::ReleaseOutputs() {
  QueuePolicy::ReleaseOutputIdxs();
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::Outputs(DeviceWorkspace *ws) {
  ReleaseOutputs();
  ShareOutputs(ws);
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::ShareOutputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(ws != nullptr, "Workspace is nullptr");
  DeviceGuard g(device_id_);
  ws->Clear();

  if (exec_error_ || QueuePolicy::IsStopSignaled()) {
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    std::string error = errors_.empty() ? "Unknown error" : errors_.front();
    throw std::runtime_error(error);
  }

  auto output_idx = QueuePolicy::UseOutputIdxs();

  if (exec_error_ || QueuePolicy::IsStopSignaled()) {
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    std::string error = errors_.empty() ? "Unknown error" : errors_.front();
    throw std::runtime_error(error);
  }

  // We already gathered info about outputs, so we only have to wait on respective
  // events to make sure that the computation has completed
  for (size_t i = 0; i < pipeline_outputs_.size(); i++) {
    auto out_tensor_id = pipeline_outputs_[i];
    auto &out_tensor = graph_->Tensor(out_tensor_id);
    auto op_type = graph_->Node(out_tensor.producer.node).op_type;
    if (out_tensor.producer.storage_device == StorageDevice::GPU) {
      VALUE_SWITCH(op_type, op_type_static, (OpType::MIXED, OpType::GPU),
      (
        auto &queue = get_queue<op_type_static, StorageDevice::GPU>(
            tensor_to_store_queue_[out_tensor_id]);
        auto stage_output_idx = output_idx[op_type_static];
        ws->AddOutput(queue[stage_output_idx]);
        CUDA_CALL(cudaEventSynchronize(gpu_output_events_[i].GetEvent(stage_output_idx)));
      ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
    } else {
      VALUE_SWITCH(op_type, op_type_static, (OpType::MIXED, OpType::GPU),
      (
        auto &queue = get_queue<op_type_static, StorageDevice::CPU>(
            tensor_to_store_queue_[out_tensor_id]);
        auto stage_output_idx = output_idx[op_type_static];
        ws->AddOutput(queue[stage_output_idx]);
      ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
    }
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::PruneUnusedGraphNodes() {
  // We want to remove any nodes whose outputs are
  // never used by another node or as an output
  DALI_ENFORCE(output_names_.size() > 0,
      "No outputs requested, nothing to execute.");

  while (true) {
    // We do not edit the graph while we are iterating
    // as node ids will be updated when an op is removed
    vector<OpNodeId> to_remove;
    for (int i = 0; i < graph_->NumOp(); ++i) {
      OpNode &node = graph_->Node(i);
      // If this node has children, don't prune it
      if (!node.children.empty()) continue;

      // Note: this is technically a very inefficient
      // way to find the intersection of the node outputs
      // and the outputs of the graph. The number of outputs
      // is usually 1-2, so it shouldn't matter
      bool found_match = false;
      for (int j = 0; j < node.spec.NumOutput(); ++j) {
        for (size_t k = 0; k < output_names_.size(); ++k) {
          if (node.spec.Output(j) == output_names_[k]) {
            found_match = true;
            break;
          }
        }
        if (found_match) break;
      }

      // If this node produces an output, don't prune it
      if (found_match) continue;

      // Mark the node for pruning
      to_remove.push_back(node.id);
    }

    // No nodes were removed, pruning complete
    if (to_remove.size() == 0) break;

    for (size_t i = 0; i < to_remove.size(); ++i) {
      // Note: After deleting a node, the graph updates
      // all other nodes in the graph to keep the node
      // ids consisten with the number of nodes in the
      // graph. 'to_remove' will store the removal
      // targets largest to smallest, so we just subtract
      // the number of previously deleted nodes from
      // the current node id.
      graph_->RemoveOp(to_remove[i] - i);
    }
  }

  // If we've pruned the entire graph, something has gone wrong
  DALI_ENFORCE(graph_->NumOp() > 0, "No output names match data produced by the pipeline.");
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::SetupOutputInfo(const OpGraph &graph) {
  DeviceGuard g(device_id_);
  pipeline_outputs_ = graph.GetOutputs(output_names_);
  for (auto tid : pipeline_outputs_) {
    auto &tensor = graph.Tensor(tid);
    DALI_ENFORCE(
        !tensor.producer.is_support,
        "Outputs of support ops cannot be outputs.");  // TODO(ptredak): lift this restriction
    if (tensor.producer.storage_device == StorageDevice::GPU) {
      auto parent_type = graph.Node(tensor.producer.node).op_type;
      gpu_output_events_.push_back(
          EventList(stage_queue_depths_[parent_type], &event_pool_));
    } else {
      gpu_output_events_.push_back(EventList());
    }
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
std::vector<int> Executor<WorkspacePolicy, QueuePolicy>::GetTensorQueueSizes(const OpGraph &graph) {
  std::vector<int> result;
  // By default we need one vector
  result.resize(graph.NumTensor(), 1);
  auto output_ids = graph.GetOutputs(output_names_);
  for (auto id : output_ids) {
    auto &tensor = graph.Tensor(id);
    auto parent_type =  graph.Node(tensor.producer.node).op_type;
    result[id] = stage_queue_depths_[parent_type];
  }
  return result;
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::PrepinData(
    std::vector<tensor_data_store_queue_t> &tensor_to_store_queue, const OpGraph &graph) {
  // We only pin what we need
  for (int i = 0; i < graph.NumOp(OpType::MIXED); i++) {
    auto &node = graph.Node(OpType::MIXED, i);
    for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
      auto tid = node.parent_tensors[j];
      // Use pinned memory only when it is useful
      if (node.spec.name() == "MakeContiguous" && node.spec.NumOutput() == 1 &&
          node.spec.OutputDevice(0) == "gpu") {
        auto &parent_tensor_queue =
            get_queue<OpType::CPU, StorageDevice::CPU>(tensor_to_store_queue_[tid]);
        for (auto &tensor : parent_tensor_queue) {
          SetPinned(tensor, true);
        }
      }
    }
  }
}

// We apply hints to all of pinned CPU buffers and all GPU buffers
template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::PresizeData(
    std::vector<tensor_data_store_queue_t> &tensor_to_store_queue, const OpGraph &graph) {
  DeviceGuard g(device_id_);
  TimeRange tr("[Executor] PresizeData");

  // To avoid handling the arguments several times for each operator that
  // has more than one output, we go over the operators instead of tensors
  for (int i = 0; i < graph.NumOp(); i++) {
    auto &node = graph.Node(i);
    auto hints = GetMemoryHints(node);
    VALUE_SWITCH(node.op_type, op_type_static,
        (OpType::SUPPORT, OpType::CPU, OpType::MIXED, OpType::GPU),
    (
      // For all tensors we produce
      for (size_t j = 0; j < node.children_tensors.size(); j++) {
        auto &tensor = graph.Tensor(node.children_tensors[j]);
        Index hint = hints[j];
        if (tensor.producer.storage_device == StorageDevice::CPU) {
          auto& queue = get_queue<op_type_static, StorageDevice::CPU>(
              tensor_to_store_queue[tensor.id]);
          for (auto storage : queue) {
            if (hint && IsPinned(storage)) {
              Reserve(storage, hint, batch_size_);
            }
          }
        } else {
          auto& queue = get_queue<op_type_static, StorageDevice::GPU>(
              tensor_to_store_queue[tensor.id]);
          for (auto storage : queue) {
            if (hint) {
              Reserve(storage, hint, batch_size_);
            }
          }
        }
      }
    ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
std::vector<int> Executor<WorkspacePolicy, QueuePolicy>::GetMemoryHints(const OpNode &node) {
  std::vector<int> hints;
  GetSingleOrRepeatedArg(node.spec, &hints, "bytes_per_sample_hint", node.spec.NumOutput());
  std::replace(hints.begin(), hints.end(), 0, static_cast<int>(bytes_per_sample_hint_));
  return hints;
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::SetupOutputQueuesForGraph() {
  QueuePolicy::InitializeQueues(stage_queue_depths_);
}

using SimpleExecutor = Executor<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;

namespace detail {

void gpu_finished_callback(cudaStream_t stream, cudaError_t status, void *userData) {
  auto callback = static_cast<ExecutorBase::ExecutorCallback*>(userData);
  (*callback)();
}

}  // namespace detail


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
