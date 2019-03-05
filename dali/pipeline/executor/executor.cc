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

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/operators/common.h"

#include "dali/pipeline/workspace/workspace_data_factory.h"
#include "dali/pipeline/graph/op_graph_storage.h"


namespace dali {


void Executor::SetCompletionCallback(ExecutorCallback cb) {
  cb_ = cb;
}

void Executor::Build(OpGraph *graph, vector<string> output_names) {
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
  tensor_to_storage_.clear();
  // Create corresponding storage type for TensorNodes in graph
  tensor_to_storage_ = CreateBackingStorageForTensorNodes(*graph_, batch_size_);
  // Setup stream and events that will be used for execution
  {
    DeviceGuard g(device_id_);
    mixed_op_stream_ = stream_pool_.GetStream();
    gpu_op_stream_ = stream_pool_.GetStream();
    mixed_op_events_ = CreateEventsForMixedOps(event_pool_, *graph_);
  }

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  WorkspaceBlob base_wsb;
  SetupWorkspacesForGraph(&base_wsb);

  // Presize the workspaces based on the hint
  PresizeData(tensor_to_storage_, *graph_);

  SetupOutputQueuesForGraph();

  // For each set of outputs, setup another set of
  // workspaces so that nothing has to be altered
  // during execution (this is necessary for
  // asynchonrous executors that can overlap work issue)
  for (int i = 0; i < queue_depth_; ++i) {
    SetOutputBuffersForIter(i, &base_wsb);
    wss_.push_back(base_wsb);
  }
}

void Executor::RunCPU() {
  TimeRange tr("[Executor] RunCPU");
  // Block until there is a free buffer to use
  std::unique_lock<std::mutex> lock(free_mutex_);
  while (free_queue_.empty() && !exec_error_) {
    free_cond_.wait(lock);
  }
  if (exec_error_) {
    return;
  }
  int queue_idx = free_queue_.front();
  free_queue_.pop();
  lock.unlock();

  DeviceGuard g(device_id_);

  // Run the support ops
  try {
    WorkspaceBlob &wsb = wss_[queue_idx];
    for (int i = 0; i < graph_->NumOp(OpType::SUPPORT); ++i) {
      OpNode &op_node = graph_->Node(OpType::SUPPORT, i);
      OperatorBase &op = *op_node.op;
      SupportWorkspace &ws = get_workspace<OpType::SUPPORT>(wsb.op_data, i);
      TimeRange tr("[Executor] Run Support op " + op_node.instance_name,
          TimeRange::kCyan);
      op.Run(&ws);
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    ready_cond_.notify_all();
  }

  if (!exec_error_) {
    // Run the cpu-ops in the thread pool
    WorkspaceBlob &wsb = wss_[queue_idx];
    for (int i = 0; i < batch_size_; ++i) {
      thread_pool_.DoWorkWithID(std::bind(
            [this, &wsb] (int data_idx, int tid) {
            TimeRange tr("[Executor] RunCPU on " + to_string(data_idx));
            SampleWorkspace ws;
            for (int j = 0; j < graph_->NumOp(OpType::CPU); ++j) {
              OpNode &op_node = graph_->Node(OpType::CPU, j);
              OperatorBase &op = *op_node.op;
              get_workspace<OpType::CPU>(wsb.op_data, op_node).GetSample(&ws, data_idx, tid);
              TimeRange tr("[Executor] Run CPU op " + op_node.instance_name
                  + " on " + to_string(data_idx),
                  TimeRange::kBlue1);
              op.Run(&ws);
            }
            }, i, std::placeholders::_1));
    }
    try {
      thread_pool_.WaitForWork();
    }
    catch (std::runtime_error& e) {
      exec_error_ = true;
      std::unique_lock<std::mutex> errors_lock(errors_mutex_);
      errors_.push_back(e.what());
      ready_cond_.notify_all();
    }
  }
  // Pass the work to the mixed stage
  std::unique_lock<std::mutex> mixed_lock(mixed_mutex_);
  mixed_work_queue_.push(queue_idx);
  mixed_lock.unlock();
}

void Executor::RunMixed() {
  TimeRange tr("[Executor] RunMixed", TimeRange::kOrange);
  std::unique_lock<std::mutex> lock(mixed_mutex_);
  DALI_ENFORCE(!mixed_work_queue_.empty(), "Mixed work "
      "queue empty. Did you call RunCPU prior to RunMixed?");
  int queue_idx = mixed_work_queue_.front();
  mixed_work_queue_.pop();
  lock.unlock();
  DeviceGuard g(device_id_);

  WorkspaceBlob &wsb = wss_[queue_idx];

  try {
    for (int i = 0; i < graph_->NumOp(OpType::MIXED); ++i) {
      OpNode &op_node = graph_->Node(OpType::MIXED, i);
      OperatorBase &op = *op_node.op;
      MixedWorkspace &ws = get_workspace<OpType::MIXED>(wsb.op_data, i);
      TimeRange tr("[Executor] Run Mixed op " + op_node.instance_name,
          TimeRange::kOrange);
      op.Run(&ws);
      if (ws.has_stream() && ws.has_event()) {
        CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
      }
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    ready_cond_.notify_all();
    free_cond_.notify_all();
  }

  // Pass the work to the gpu stage
  std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
  gpu_work_queue_.push(queue_idx);
  gpu_lock.unlock();
}

void Executor::RunGPU() {
  TimeRange tr("[Executor] RunGPU", TimeRange::knvGreen);
  std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
  DALI_ENFORCE(!gpu_work_queue_.empty(), "GPU work queue "
      "empty. Did you call RunMixed prior to RunGPU?");
  int queue_idx = gpu_work_queue_.front();
  gpu_work_queue_.pop();
  gpu_lock.unlock();
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
    WorkspaceBlob &wsb = wss_[queue_idx];
    for (int i = 0; i < graph_->NumOp(OpType::GPU); ++i) {
      OpNode &op_node = graph_->Node(OpType::GPU, i);
      OperatorBase &op = *op_node.op;
      DeviceWorkspace &ws = get_workspace<OpType::GPU>(wsb.op_data, i);
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

    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (graph_->TensorIsType<CPUBackend>(output_names_[i])) continue;
      OpNodeId src_id = graph_->TensorSourceID(output_names_[i]);
      int src_idx = graph_->NodeIdx(src_id);

      // Record events for each output requested by the user
      cudaEvent_t event = gpu_output_events_[i].GetEvent(queue_idx);
      if (graph_->NodeType(src_id) == OpType::MIXED) {
        auto &ws = get_workspace<OpType::MIXED>(wsb.op_data, src_idx);
        CUDA_CALL(cudaEventRecord(event, ws.stream()));
      } else if (graph_->NodeType(src_id) == OpType::GPU) {
        auto &ws = get_workspace<OpType::GPU>(wsb.op_data, src_idx);
        CUDA_CALL(cudaEventRecord(event, ws.stream()));
      } else {
        DALI_FAIL("Internal error. Output node is not gpu/mixed");
      }
    }
  } catch (std::runtime_error &e) {
    exec_error_ = true;
    std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    errors_.push_back(e.what());
    free_cond_.notify_all();
    ready_cond_.notify_all();
    return;
  }
  // Update the ready queue to signal that all the work
  // in the `queue_idx` set of output buffers has been
  // issued. Notify any waiting threads.
  std::unique_lock<std::mutex> lock(ready_mutex_);
  ready_queue_.push(queue_idx);
  ready_cond_.notify_all();
  lock.unlock();

  // Save the queue_idx so we can enforce the
  // dependency between consecutive iterations
  // of the gpu stage of the pipeline.
  previous_gpu_queue_idx_ = queue_idx;

  // call any registered previously callback
  if (cb_) {
    cb_();
  }
}

void Executor::ReleaseOutputs() {
  // Mark the last in-use buffer as free and signal
  // to waiting threads
  if (!in_use_queue_.empty()) {
    std::unique_lock<std::mutex> lock(free_mutex_);
    free_queue_.push(in_use_queue_.front());
    in_use_queue_.pop();
    free_cond_.notify_one();
    lock.unlock();
  }
}

void Executor::Outputs(DeviceWorkspace *ws) {
  ReleaseOutputs();
  ShareOutputs(ws);
}

void Executor::ShareOutputs(DeviceWorkspace *ws) {
  DALI_ENFORCE(ws != nullptr, "Workspace is nullptr");
  DeviceGuard g(device_id_);
  ws->Clear();

  if (exec_error_) {
    std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    std::string error = errors_.empty() ? "Unknown error" : errors_.front();
    throw std::runtime_error(error);
  }

  // Block until the work for a batch has been issued.
  // Move the queue id from ready to in_use
  std::unique_lock<std::mutex> lock(ready_mutex_);
  while (ready_queue_.empty() && !exec_error_) {
    ready_cond_.wait(lock);
    if (exec_error_) {
      break;
    }
  }
  if (exec_error_) {
    std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    std::string error = errors_.empty() ? "Unknown error" : errors_.front();
    throw std::runtime_error(error);
  }
  int output_idx = ready_queue_.front();
  ready_queue_.pop();
  in_use_queue_.push(output_idx);
  lock.unlock();

  // Gather the results TensorLists and block on their
  // events to make sure that the computation has completed
  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto it = type_idx_map_.find(output_names_[i]);
    DALI_ENFORCE(it != type_idx_map_.end(), "Executor could not "
        "find output with name '" + output_names_[i] + "'.");

    if (graph_->TensorIsType<CPUBackend>(output_names_[i])) {
      auto &tl_pool = cpu_outputs_[it->second];
      ws->AddOutput(tl_pool.Get(output_idx));
    } else {
      auto &tl_pool = gpu_outputs_[it->second];
      ws->AddOutput(tl_pool.Get(output_idx));
      CUDA_CALL(cudaEventSynchronize(
              gpu_output_events_[i].GetEvent(output_idx)));
    }
  }
}

void Executor::PruneUnusedGraphNodes() {
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
  DALI_ENFORCE(graph_->NumOp() > 0, "No output names match "
      "data produced by the pipeline.");
}

void Executor::SetupWorkspacesForGraph(WorkspaceBlob *wsb) {
  DeviceGuard g(device_id_);

  // Clear any old data setup
  wsb->Clear();
  wsb->Resize(graph_->NumOp(OpType::SUPPORT), graph_->NumOp(OpType::CPU),
              graph_->NumOp(OpType::MIXED), graph_->NumOp(OpType::GPU));

  for (int i = 0; i < graph_->NumOp(); i++) {
    auto &node = graph_->Node(i);
    VALUE_SWITCH(node.op_type, op_type_static,
        (OpType::SUPPORT, OpType::CPU, OpType::MIXED, OpType::GPU),
    (
      auto &ws = get_workspace<op_type_static>(wsb->op_data, node);
      ws = CreateWorkspace<op_type_static>(*graph_, node);
    ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
  }
}

// We apply hints to all of pinned CPU buffers and all GPU buffers
void Executor::PresizeData(std::vector<tensor_data_store_t> &tensor_to_storage,
                           const OpGraph &graph) {
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
          auto storage = get_storage<op_type_static, StorageDevice::CPU>(
              tensor_to_storage[tensor.id]);
          if (hint && IsPinned(storage)) {
            Reserve(storage, hint, batch_size_);
          }
        } else {
          auto storage = get_storage<op_type_static, StorageDevice::GPU>(
              tensor_to_storage[tensor.id]);
          if (hint) {
            Reserve(storage, hint, batch_size_);
          }
        }
      }
    ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
  }
}

std::vector<int> Executor::GetMemoryHints(const OpNode &node) {
  std::vector<int> hints;
  GetSingleOrRepeatedArg(node.spec, &hints, "bytes_per_sample_hint", node.spec.NumOutput());
  std::replace(hints.begin(), hints.end(), 0, static_cast<int>(bytes_per_sample_hint_));
  return hints;
}

void Executor::SetupOutputQueuesForGraph() {
  DeviceGuard g(device_id_);
  // Allocate output TensorList pools for each output
  for (auto &name : output_names_) {
    auto tensor_meta = graph_->TensorSourceMeta(name);

    // TODO(klecki): this is a duplication of TensorNode functionality here in convoluted manner
    // Collect meta-data about the tensor for fast lookup later.
    OutputInfo info;
    info.prod_and_idx = std::make_pair(tensor_meta.node, tensor_meta.index);
    vector<TensorMeta> consumer_meta = graph_->TensorConsumerMeta(name);
    for (auto &meta : consumer_meta) {
      auto tmp = std::make_pair(meta.node, meta.index);
      info.con_and_idx.push_back(tmp);
    }

    // Create the buffer and events
    if (tensor_meta.storage_device == StorageDevice::CPU) {
      DALI_ENFORCE(
          !tensor_meta.is_support,
          "Outputs of support ops cannot be outputs.");  // TODO(ptredak): lift this restriction
      cpu_outputs_.push_back(
          TensorListPool<CPUBackend>(queue_depth_, batch_size_, bytes_per_sample_hint_));
      DALI_ENFORCE(
          type_idx_map_.insert({name, cpu_outputs_.size() - 1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" + name + "' exists.");

      cpu_output_info_.push_back(info);
      gpu_output_events_.push_back(EventList());
    } else {
      gpu_outputs_.push_back(
          TensorListPool<GPUBackend>(queue_depth_, batch_size_, bytes_per_sample_hint_));
      DALI_ENFORCE(
          type_idx_map_.insert({name, gpu_outputs_.size() - 1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" + name + "' exists.");

      gpu_output_info_.push_back(info);
      gpu_output_events_.push_back(EventList(queue_depth_, &event_pool_));
    }
  }

  // All buffers start off as free
  for (int i = 0; i < queue_depth_; ++i) {
    free_queue_.push(i);
  }
}

void Executor::SetOutputBuffersForIter(int queue_idx, WorkspaceBlob *wsb) {
  DeviceGuard g(device_id_);
  // For each output, we need to hookup the next buffer
  // to the desired output workspaces, and also the
  // input workspaces of later ops that use them
  for (size_t i = 0; i < cpu_outputs_.size(); ++i) {
    auto &info = cpu_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;
    // Contiguous CPU outputs come from mixed or GPU ops
    DALI_ENFORCE(graph_->NodeType(node_id) == OpType::MIXED ||
                 graph_->NodeType(node_id) == OpType::GPU);

    if (graph_->NodeType(node_id) == OpType::MIXED) {
      int mixed_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::MIXED>(wsb->op_data, mixed_op_id)
          .SetOutput(output_idx, cpu_outputs_[i].Get(queue_idx));
    } else {  // OpType::GPU
      int gpu_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::GPU>(wsb->op_data, gpu_op_id)
          .SetOutput(output_idx, cpu_outputs_[i].Get(queue_idx));
    }

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      DALI_ENFORCE(graph_->NodeType(node_id) == OpType::GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::GPU>(wsb->op_data, gpu_op_id)
          .SetInput(input_idx, cpu_outputs_[i].Get(queue_idx));
    }
  }

  for (size_t i = 0; i < gpu_outputs_.size(); ++i) {
    auto &info = gpu_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;

    if (graph_->NodeType(node_id) == OpType::MIXED) {
      int mixed_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::MIXED>(wsb->op_data, mixed_op_id)
          .SetOutput(output_idx, gpu_outputs_[i].Get(queue_idx));
    } else if (graph_->NodeType(node_id) == OpType::GPU) {
      int gpu_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::GPU>(wsb->op_data, gpu_op_id)
          .SetOutput(output_idx, gpu_outputs_[i].Get(queue_idx));
    } else {
      DALI_FAIL("Internal error. GPU output source is not gpu/mixed op");
    }

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      DALI_ENFORCE(graph_->NodeType(node_id) == OpType::GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      get_workspace<OpType::GPU>(wsb->op_data, gpu_op_id)
          .SetInput(input_idx, gpu_outputs_[i].Get(queue_idx));
    }
  }
}

}  // namespace dali
