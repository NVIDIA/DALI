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

#include "dali/pipeline/executor/executor.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dali {


void Executor::SetCompletionCallback(ExecutorCallback cb) {
  cb_ = cb;
}

void Executor::Build(OpGraph *graph, vector<string> output_names) {
  DALI_ENFORCE(graph != nullptr, "Input graph is nullptr.");
  DALI_ENFORCE(graph->NumOp() > 0, "Graph has no operators.");
  output_names_ = output_names;
  graph_ = graph;

  DeviceGuard g(device_id_);

  // Remove any node from the graph whose output
  // will not be used as an output or by another node
  PruneUnusedGraphNodes();

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  WorkspaceBlob base_wsb;
  SetupDataForGraph(&base_wsb);

  // Presize the workspaces based on the hint
  PresizeData(&base_wsb);

  // Assign streams to all mixed & gpu ops
  SetupStreamsForGraph(&base_wsb);

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
    for (int i = 0; i < graph_->NumSupportOp(); ++i) {
      OpNode &op_node = graph_->support_node(i);
      OperatorBase &op = *op_node.op;
      SupportWorkspace &ws = wsb.support_op_data[i];
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
            for (int j = 0; j < graph_->NumCPUOp(); ++j) {
              OpNode &op_node = graph_->cpu_node(j);
              OperatorBase &op = *op_node.op;
              wsb.cpu_op_data[j].GetSample(&ws, data_idx, tid);
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
  TimeRange tr("[Executor] RunMixed");
  std::unique_lock<std::mutex> lock(mixed_mutex_);
  DALI_ENFORCE(!mixed_work_queue_.empty(), "Mixed work "
      "queue empty. Did you call RunCPU prior to RunMixed?");
  int queue_idx = mixed_work_queue_.front();
  mixed_work_queue_.pop();
  lock.unlock();
  DeviceGuard g(device_id_);

  WorkspaceBlob &wsb = wss_[queue_idx];

  try {
    for (int i = 0; i < graph_->NumMixedOp(); ++i) {
      OpNode &op_node = graph_->mixed_node(i);
      OperatorBase &op = *op_node.op;
      MixedWorkspace &ws = wsb.mixed_op_data[i];
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
  TimeRange tr("[Executor] RunGPU");
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
    for (int i = 0; i < graph_->NumGPUOp(); ++i) {
      OpNode &op_node = graph_->gpu_node(i);
      OperatorBase &op = *op_node.op;
      DeviceWorkspace &ws = wsb.gpu_op_data[i];
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
      NodeID src_id = graph_->TensorSourceID(output_names_[i]);
      int src_idx = graph_->NodeIdx(src_id);

      // Record events for each output requested by the user
      cudaEvent_t event = gpu_output_events_[i].GetEvent(queue_idx);
      if (graph_->NodeType(src_id) == DALI_MIXED) {
        auto &ws = wsb.mixed_op_data[src_idx];
        CUDA_CALL(cudaEventRecord(event, ws.stream()));
      } else if (graph_->NodeType(src_id) == DALI_GPU) {
        auto &ws = wsb.gpu_op_data[src_idx];
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
    vector<NodeID> to_remove;
    for (int i = 0; i < graph_->NumOp(); ++i) {
      OpNode &node = graph_->node(i);
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

void Executor::SetupDataForGraph(WorkspaceBlob *wsb) {
  DeviceGuard g(device_id_);

  // Clear any old data setup
  wsb->Clear();

  // Create workspaces for each operator
  wsb->cpu_op_data.resize(graph_->NumCPUOp());
  wsb->mixed_op_data.resize(graph_->NumMixedOp());
  wsb->gpu_op_data.resize(graph_->NumGPUOp());
  wsb->support_op_data.resize(graph_->NumSupportOp());

  // Setup support op input and output buffers
  for (int i = 0; i < graph_->NumSupportOp(); ++i) {
    OpNode &node = graph_->support_node(i);
    SupportWorkspace &ws = wsb->support_op_data[i];
    // Support ops do not take argument inputs
    DALI_ENFORCE(node.spec.NumInput() == node.spec.NumRegularInput(),
        "Support ops do not support tensor arguments");
    for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
      // Get each regular input and add them to this op's workspace.

      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      DALI_ENFORCE(parent_op_type == DALI_SUPPORT,
          "Executor encountered support op with non-support input.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      SupportWorkspace &src_ws = wsb->support_op_data[parent_idx];
      const auto input = src_ws.SharedCPUOutput(input_src_idx);
      ws.AddInput(input);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // Allocate tensors for output
      shared_ptr<Tensor<CPUBackend>> output(new Tensor<CPUBackend>);
      output->set_pinned(false);
      ws.AddOutput(output);
    }
  }

  // Setup cpu op input and output buffers
  for (int i = 0; i < graph_->NumCPUOp(); ++i) {
    OpNode &node = graph_->cpu_node(i);
    HostWorkspace &ws = wsb->cpu_op_data[i];
    for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
      // Get each regular input and add them to this op's workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      DALI_ENFORCE(parent_op_type == DALI_CPU,
          "Executor encountered cpu op with non-cpu input.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = wsb->cpu_op_data[parent_idx];
      const auto input = src_ws.SharedCPUOutput(input_src_idx);
      ws.AddInput(input);
    }

    // Add argument tensors
    for (const auto &arg_pair : node.spec.ArgumentInputs()) {
      // Get each argument input and add them to this op's workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(arg_pair.second));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      DALI_ENFORCE(parent_op_type == DALI_SUPPORT,
          "Executor encountered argument input produced by non-cpu op.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(arg_pair.second));

      SupportWorkspace &src_ws = wsb->support_op_data[parent_idx];
      const auto input = src_ws.SharedCPUOutput(input_src_idx);
      ws.AddArgumentInput(input, arg_pair.first);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // Allocate `batch_size` Tensors for this ops
      // results and add them to the workspace.
      vector<shared_ptr<Tensor<CPUBackend>>> output(batch_size_, nullptr);
      for (auto &tensor_ptr : output) {
        tensor_ptr.reset(new Tensor<CPUBackend>);
        tensor_ptr->set_pinned(false);
      }

      ws.AddOutput(output);
    }
  }

  // Setup mixed op input and output buffers
  for (int i = 0; i < graph_->NumMixedOp(); ++i) {
    OpNode &node = graph_->mixed_node(i);
    MixedWorkspace &ws = wsb->mixed_op_data[i];

    // Mixed ops do not take argument tensors (at least for now)
    DALI_ENFORCE(node.spec.NumRegularInput() == node.spec.NumInput());

    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this mixed ops workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      DALI_ENFORCE(parent_op_type == DALI_CPU,
          "Executor encountered mixed op with non-cpu input.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = wsb->cpu_op_data[parent_idx];
      auto input = src_ws.SharedCPUOutput(input_src_idx);
      // Use pinned memory only when it is useful
      if (node.spec.name() == "MakeContiguous" &&
          node.spec.NumOutput() == 1 &&
          node.spec.OutputDevice(0) == "gpu") {
        for (auto t : input) {
          t->set_pinned(true);
        }
      }
      ws.AddInput(input);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      if (node.spec.OutputDevice(j) == "cpu") {
        // Allocate TensorLists for this ops outputs
        ws.AddOutput(std::make_shared<TensorList<CPUBackend>>());
      } else if (node.spec.OutputDevice(j) == "gpu") {
        ws.AddOutput(std::make_shared<TensorList<GPUBackend>>());
      } else {
        DALI_FAIL("Executor encountered mixed op with non-gpu/cpu output.");
      }
    }
  }

  // Setup gpu op input and output buffers
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    OpNode &node = graph_->gpu_node(i);
    DeviceWorkspace &ws = wsb->gpu_op_data[i];
    for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
      // Get each input and add them to this GPU op's workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      if (parent_op_type == DALI_MIXED) {
        MixedWorkspace &src_ws = wsb->mixed_op_data[parent_idx];
        if (node.spec.InputDevice(j) == "cpu") {
          const auto input = src_ws.SharedCPUOutput(input_src_idx);
          ws.AddInput(input);
        } else if (node.spec.InputDevice(j) == "gpu") {
          const auto input = src_ws.SharedGPUOutput(input_src_idx);
          ws.AddInput(input);
        } else {
          DALI_FAIL("Executor encountered gpu op with non-cpu/gpu input.");
        }
      } else if (parent_op_type == DALI_GPU) {
        DeviceWorkspace &src_ws = wsb->gpu_op_data[parent_idx];
        if (node.spec.InputDevice(j) == "cpu") {
          // Note: This path should currently never occur, as we
          // do not allow gpu ops to produce cpu data outputs.
          const auto input = src_ws.SharedCPUOutput(input_src_idx);
          ws.AddInput(input);
        } else if (node.spec.InputDevice(j) == "gpu") {
          const auto input = src_ws.SharedGPUOutput(input_src_idx);
          ws.AddInput(input);
        } else {
          DALI_FAIL("Executor encountered gpu op with non-cpu/gpu input.");
        }
      } else {
        DALI_FAIL("Executor encountered gpu op with non-mixed/gpu input.");
      }
    }

    // Add argument tensors
    for (const auto &arg_pair : node.spec.ArgumentInputs()) {
      // Get each argument input and add them to this op's workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(arg_pair.second));
      DALIOpType parent_op_type = graph_->NodeType(parent_node_id);
      DALI_ENFORCE(parent_op_type == DALI_SUPPORT,
          "Executor encountered argument input produced by non-cpu op.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(arg_pair.second));

      SupportWorkspace &src_ws = wsb->support_op_data[parent_idx];
      auto input = src_ws.SharedCPUOutput(input_src_idx);
      input->set_pinned(true);
      ws.AddArgumentInput(input, arg_pair.first);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // Allocate TensorLists for this ops output
      if (node.spec.OutputDevice(j) == "gpu") {
        ws.AddOutput(std::make_shared<TensorList<GPUBackend>>());
      } else if (node.spec.OutputDevice(j) == "cpu") {
        ws.AddOutput(std::make_shared<TensorList<CPUBackend>>());
      } else {
        DALI_FAIL("Executor encountered gpu op with non cpu/gpu output.");
      }
    }
  }
}

void Executor::PresizeData(WorkspaceBlob *wsb) {
  TimeRange tr("[Executor] PresizeData");
  DeviceGuard g(device_id_);
  // Note: At some point our graph has source nodes that
  // only have outputs (data readers or external inputs).
  // Thus, the set of all outputs buffers in our workspaces
  // represents all the unique buffers in our graph.
  for (auto &ws : wsb->cpu_op_data) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      DALI_ENFORCE(ws.NumOutputAtIdx(i) == batch_size_, "Executor "
          "encountered cpu op workspace where the number of tensors "
          "is not equal to the batch size.");
      DALI_ENFORCE(ws.OutputIsType<CPUBackend>(i), "Executor "
          "encountered cpu op with non-cpu output.");
      for (int j = 0; j < ws.NumOutputAtIdx(i); ++j) {
        Tensor<CPUBackend> *tensor = ws.Output<CPUBackend>(i, j);
        // We set the type of the tensor to uint8 temporarily
        tensor->mutable_data<uint8>();
        tensor->Resize({(Index)bytes_per_sample_hint_});
      }
    }
  }

  for (auto &ws : wsb->mixed_op_data) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      if (ws.OutputIsType<CPUBackend>(i)) {
        TensorList<CPUBackend> *tl = ws.Output<CPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      } else {
        TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      }
    }
  }

  for (auto &ws : wsb->gpu_op_data) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      if (ws.OutputIsType<GPUBackend>(i)) {
        TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      } else {
        TensorList<CPUBackend> *tl = ws.Output<CPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      }
    }
  }
}

void Executor::SetupStreamsForGraph(WorkspaceBlob *wsb) {
  DeviceGuard g(device_id_);
  auto mixed_op_stream = stream_pool_.GetStream();
  for (int i = 0; i < graph_->NumMixedOp(); ++i) {
    // We assign unique stream to mixed ops.
    // This ensures that we won't have false dependencies
    // between mixed ops and the previous iterations
    // gpu ops.
    MixedWorkspace &ws = wsb->mixed_op_data[i];
    ws.set_stream(mixed_op_stream);
    ws.set_event(event_pool_.GetEvent());
  }

  // I/O pipeline is always going to be launched alongside
  // some other GPU work (like DL training).
  // Therefore it is not necessary to use more than
  // 1 stream for GPU ops, even though we may not fill
  // the whole GPU with just I/O pipeline kernels
  // by doing so.
  auto gpu_op_stream = stream_pool_.GetStream();
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    DeviceWorkspace &ws = wsb->gpu_op_data[i];
    ws.set_stream(gpu_op_stream);
    const OpNode& node = graph_->gpu_node(i);
    for (const auto& p : node.parents) {
      if (graph_->NodeType(p) == DALI_MIXED) {
        // We need to block on this op's event to
        // make sure that we respect the dependency
        int parent_op_idx = graph_->NodeIdx(p);
        MixedWorkspace parent_ws = wsb->mixed_op_data[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      }
    }
  }
}

void Executor::SetupOutputQueuesForGraph() {
  DeviceGuard g(device_id_);
  // Allocate output TensorList pools for each output
  for (auto &name : output_names_) {
    auto tensor_meta = graph_->TensorSourceMeta(name);

    // Collect meta-data about the tensor for fast lookup later.
    OutputInfo info;
    info.prod_and_idx = std::make_pair(tensor_meta.node, tensor_meta.index);
    vector<TensorMeta> consumer_meta = graph_->TensorConsumerMeta(name);
    for (auto &meta : consumer_meta) {
      auto tmp = std::make_pair(meta.node, meta.index);
      info.con_and_idx.push_back(tmp);
    }

    // Create the buffer and events
    if (tensor_meta.is_cpu) {
      DALI_ENFORCE(!tensor_meta.is_support,
          "Outputs of support ops cannot be outputs.");  // TODO(ptredak): lift this restriction
      cpu_outputs_.push_back(TensorListPool<CPUBackend>(
              queue_depth_, batch_size_, bytes_per_sample_hint_));
      DALI_ENFORCE(type_idx_map_.insert({name, cpu_outputs_.size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");

      cpu_output_info_.push_back(info);
      gpu_output_events_.push_back(EventList());
    } else {
      gpu_outputs_.push_back(TensorListPool<GPUBackend>(
              queue_depth_, batch_size_, bytes_per_sample_hint_));
      DALI_ENFORCE(type_idx_map_.insert({name, gpu_outputs_.size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");

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
    NodeID node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;
    // Contiguous CPU outputs come from mixed or GPU ops
    DALI_ENFORCE(graph_->NodeType(node_id) == DALI_MIXED ||
                 graph_->NodeType(node_id) == DALI_GPU);

    if (graph_->NodeType(node_id) == DALI_MIXED) {
      int mixed_op_id = graph_->NodeIdx(node_id);
      wsb->mixed_op_data[mixed_op_id].SetOutput(
          output_idx, cpu_outputs_[i].Get(queue_idx));
    } else {  // DALI_GPU
      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetOutput(output_idx,
          cpu_outputs_[i].Get(queue_idx));
    }

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      DALI_ENFORCE(graph_->NodeType(node_id) == DALI_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetInput(
          input_idx, cpu_outputs_[i].Get(queue_idx));
    }
  }

  for (size_t i = 0; i < gpu_outputs_.size(); ++i) {
    auto &info = gpu_output_info_[i];
    NodeID node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;

    if (graph_->NodeType(node_id) == DALI_MIXED) {
      int mixed_op_id = graph_->NodeIdx(node_id);
      wsb->mixed_op_data[mixed_op_id].SetOutput(output_idx,
          gpu_outputs_[i].Get(queue_idx));
    } else if (graph_->NodeType(node_id) == DALI_GPU) {
      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetOutput(output_idx,
          gpu_outputs_[i].Get(queue_idx));
    } else {
      DALI_FAIL("Internal error. GPU output source is "
          "not gpu/mixed op");
    }

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      DALI_ENFORCE(graph_->NodeType(node_id) == DALI_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetInput(input_idx,
          gpu_outputs_[i].Get(queue_idx));
    }
  }
}

}  // namespace dali
