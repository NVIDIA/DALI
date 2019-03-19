// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_WORKSPACE_POLICY_H_
#define DALI_PIPELINE_EXECUTOR_WORKSPACE_POLICY_H_

#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

// We instantiate the operation of adding the input only for parent op_type and device
// that are specifically allowed
// We always use queue_idx = 0 if give queue has only one element -> it is not queued
template <OpType op_type, OpType producer_type, StorageDevice device>
enable_if_t<allows_op_input<op_type>(producer_type) && allows_tensor_input<op_type>(device)>
add_input(op_type_to_workspace_t<op_type> &ws, const tensor_data_store_queue_t &storage,
          int queue_idx = 0) {
  auto &queue = get_queue<producer_type, device>(storage);
  DALI_ENFORCE(!queue.IsBuffered() || queue_idx < static_cast<int>(queue.size()),
               "Backing Tensor store queue has not enough elements.");
  auto tensor = queue[queue_idx];
  ws.AddInput(tensor);
}

// If parent op_type or device is not allowed this is a no-op
template <OpType op_type, OpType producer_type, StorageDevice device>
enable_if_t<!allows_op_input<op_type>(producer_type) || !allows_tensor_input<op_type>(device)>
add_input(op_type_to_workspace_t<op_type>, const tensor_data_store_queue_t, int = 0) {}

template <OpType op_type, StorageDevice device>
void add_output(op_type_to_workspace_t<op_type> &ws, const tensor_data_store_queue_t &storage,
                int queue_idx = 0) {
  auto &queue = get_queue<op_type, device>(storage);
  DALI_ENFORCE(!queue.IsBuffered() || queue_idx < static_cast<int>(queue.size()),
               "Backing Tensor store queue has not enough elements.");
  auto tensor = queue[queue_idx];
  ws.AddOutput(tensor);
}

// TODO(klecki): should we move this to OpNode, and make it subclasses for
// all DALIOpTypes -> implement `add_input` and add_output for all of the subclasses
// as well with later operations
template <OpType op_type>
void SetupInputOutput(op_type_to_workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
                      const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                      const QueueIdxs idxs) {
  for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
    auto tid = node.parent_tensors[j];
    auto &parent_node = graph.Node(graph.Tensor(tid).producer.node);
    auto parent_op_type = parent_node.op_type;
    auto tensor_device = graph.Tensor(tid).producer.storage_device;

    VALUE_SWITCH(parent_op_type, parent_op_static,
        (OpType::GPU, OpType::CPU, OpType::MIXED, OpType::SUPPORT),
    (
      VALUE_SWITCH(tensor_device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
      (
        add_input<op_type, parent_op_static, device_static>(ws,
                                                            tensor_to_store_queue[tid],
                                                            idxs[parent_op_static]);
      ), DALI_FAIL("Unexpected device"))  // NOLINT(whitespace/parens)
    ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
  }

  // Argument inputs can be handled genericaly
  for (const auto &arg_pair : node.spec.ArgumentInputs()) {
    // Get each argument input and add them to this op's workspace.
    auto input_index = arg_pair.second;
    auto tid = node.parent_tensors[input_index];
    // Argument inputs are only CPU
    auto &queue = get_queue<OpType::SUPPORT, StorageDevice::CPU>(tensor_to_store_queue[tid]);
    auto tensor = queue[idxs[OpType::SUPPORT]];
    ws.AddArgumentInput(tensor, arg_pair.first);
  }

  for (int j = 0; j < node.spec.NumOutput(); ++j) {
    auto tid = node.children_tensors[j];
    auto tensor_device = graph.Tensor(tid).producer.storage_device;
    VALUE_SWITCH(tensor_device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
    (
      add_output<op_type, device_static>(ws, tensor_to_store_queue[tid], idxs[op_type]);
    ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
  }
}

template <OpType op_type>
void SetupStreamsAndEvents(op_type_to_workspace_t<op_type> &ws, const OpGraph &graph,
                           const OpNode &node, cudaStream_t mixed_op_stream,
                           cudaStream_t gpu_op_stream,
                           const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                           const QueueIdxs idxs) {
  /* No-op if we are not Mixed or GPU */
}

template <>
inline void SetupStreamsAndEvents<OpType::MIXED>(
    MixedWorkspace &ws, const OpGraph &graph, const OpNode &node, cudaStream_t mixed_op_stream,
    cudaStream_t gpu_op_stream, const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
    const QueueIdxs idxs) {
  // We assign unique stream to mixed ops.
  // This ensures that we won't have false dependencies
  // between mixed ops and the previous iterations
  // gpu ops.
  ws.set_stream(mixed_op_stream);
  ws.set_event(mixed_op_events[node.partition_index][idxs[OpType::MIXED]]);
}

template <>
inline void SetupStreamsAndEvents<OpType::GPU>(
    DeviceWorkspace &ws, const OpGraph &graph, const OpNode &node, cudaStream_t mixed_op_stream,
    cudaStream_t gpu_op_stream, const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
    const QueueIdxs idxs) {
  // I/O pipeline is always going to be launched alongside
  // some other GPU work (like DL training).
  // Therefore it is not necessary to use more than
  // 1 stream for GPU ops, even though we may not fill
  // the whole GPU with just I/O pipeline kernels
  // by doing so.
  ws.set_stream(gpu_op_stream);
  for (const auto &p : node.parents) {
    if (graph.NodeType(p) == OpType::MIXED) {
      const auto &parent_op = graph.Node(p);
      // We need to block on this op's event to
      // make sure that we respect the dependency
      ws.AddParentEvent(mixed_op_events[parent_op.partition_index][idxs[OpType::MIXED]]);
    }
  }
}

template <OpType op_type>
op_type_to_workspace_t<op_type> CreateWorkspace(
    const OpGraph &graph, const OpNode &node,
    const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
    cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
    const std::vector<std::vector<cudaEvent_t>> &mixed_op_events, const QueueIdxs idxs) {
  op_type_to_workspace_t<op_type> ws;
  SetupInputOutput<op_type>(ws, graph, node, tensor_to_store_queue, idxs);
  // SetupPinned<op_type>(ws, graph, node, idxs);
  SetupStreamsAndEvents<op_type>(ws, graph, node, mixed_op_stream, gpu_op_stream, mixed_op_events,
                                 idxs);
  return ws;
}

/**
 * @brief Policy that is responsible for providing executor with workspaces used
 * during RunX() functions.
 */
// template <typename QueuePolicy>
// struct WS_Policy {
//   // Type trait describing how will the workspace be returned (usually by copy or by ref)
//   template <OpType op_type>
//   using ws_t = ...;
//
//   // Initialize state of Workspace Storage
//   void InitializeWorkspaceStore(const OpGraph &graph,
//                                 const std::vector<tensor_data_store_queue_t>
//                                 &tensor_to_store_queue, cudaStream_t mixed_op_stream,
//                                 cudaStream_t gpu_op_stream, const
//                                 std::vector<std::vector<cudaEvent_t>> &mixed_op_events, const
//                                 QueueSizes idxs);
//   /**
//    * @brief Get the Workpsace for given `op_type` stage, when executing queue indexes `idx` part
//    * of job, needed to execute node with `partition_idx` in stage `op_type`.
//    * @tparam op_type Stage
//    * @param idxs
//    * @param graph
//    * @param partition_idx Index of the OpNode in its `op_type` partition of graph
//    * @return Corresponding workspace for that operator/OpNode
//    */
//   template <OpType op_type>
//   ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx);
//
//   /**
//    * @brief Get the Workpsace for given `op_type` stage, when executing queue indexes `idx` part
//    * of job, needed to execute the `node`.
//    *
//    * @tparam op_type
//    * @param idxs
//    * @param graph
//    * @param node OpNode for which we return the workspac
//    * @return Corresponding workspace for that operator/OpNode
//    */
//   template <OpType op_type>
//   ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node);
// };

/**
 * @brief Just In Time Workspace Policy
 *
 * When requested, returns a copy of a new workspaces, filling it on the fly with
 * existing inputs, outputs, streams and events.
 *
 * Intended to be used with SeparateQueuePolicy
 */
struct JIT_WS_Policy {
  template <OpType op_type>
  using ws_t = op_type_to_workspace_t<op_type>;

  void InitializeWorkspaceStore(const OpGraph &graph,
                                const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                                cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
                                const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                                const QueueSizes idxs) {
    tensor_to_store_queue_ = tensor_to_store_queue;
    mixed_op_stream_ = mixed_op_stream;
    gpu_op_stream_ = gpu_op_stream;
    mixed_op_events_ = mixed_op_events;
    queue_sizes_ = idxs;
  }

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
    return CreateWorkspace<op_type>(graph, graph.Node(op_type, partition_idx),
                                    tensor_to_store_queue_, mixed_op_stream_, gpu_op_stream_,
                                    mixed_op_events_, idxs);
  }

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
    DALI_ENFORCE(node.op_type == op_type,
                 "Wrong variant of method selected. OpType does not match.");
    return CreateWorkspace<op_type>(graph, node, tensor_to_store_queue_, mixed_op_stream_,
                                    gpu_op_stream_, mixed_op_events_, idxs);
  }

 private:
  // TODO(klecki): should consider if storing copy of backing storage is good idea
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  std::vector<std::vector<cudaEvent_t>> mixed_op_events_;
  QueueSizes queue_sizes_;
};

inline int SequentialIndex(QueueIdxs idxs, QueueIdxs max_idxs, OpType last_stage) {
  constexpr static OpType order[] = {OpType::SUPPORT, OpType::CPU, OpType::MIXED, OpType::GPU};
  // Horner's method
  int result = 0;
  for (auto op_type : order) {
    result += idxs[op_type];
    if (op_type == last_stage) {
      return result;
    }
    result *= max_idxs[op_type];
  }
  DALI_FAIL("Error when calculating index - unexpected stage");
}


/**
 * @brief AOT2 Workspace Policy
 *
 * Intended to be used with SeparateQueuePolicy, creates all workspaces ahead of time
 */
struct AOT2_WS_Policy {
  AOT2_WS_Policy() : depths_(0) {}

  template <OpType op_type>
  using ws_t = op_type_to_workspace_t<op_type> &;

  void InitializeWorkspaceStore(const OpGraph &graph,
                                const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                                cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
                                const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                                const QueueSizes idxs) {
    // Tie Mixed and GPU together
    // TODO(klecki): propagate the queue sizes from executor here

    depths_[OpType::SUPPORT] = idxs.cpu_size * idxs.gpu_size;
    depths_[OpType::CPU] = idxs.cpu_size;
    // Mixed and GPU are bound together due to being outputs
    depths_[OpType::MIXED] = idxs.gpu_size;
    depths_[OpType::GPU] = idxs.gpu_size;
    support_workspaces_.resize(SequentialIndex(depths_, depths_, OpType::SUPPORT));
    cpu_workspaces_.resize(SequentialIndex(depths_, depths_, OpType::CPU));
    // We do not create another layer of ws, as we do Mixed and GPU together
    mixed_workspaces_.resize(SequentialIndex(depths_, depths_, OpType::MIXED));
    gpu_workspaces_.resize(SequentialIndex(depths_, depths_, OpType::MIXED));

    // now we do cover possible calls for GetWorkspace - for all possible QueueIdxs that we may get
    for (int support_id = 0; support_id < depths_[OpType::SUPPORT]; support_id++) {
      // Get sequential index and prepare space all Support Ops
      auto queue_idxs = QueueIdxs{support_id, 0, 0, 0};
      PlaceWorkspace<OpType::SUPPORT>(support_workspaces_, queue_idxs, graph, tensor_to_store_queue,
                                      mixed_op_stream, gpu_op_stream, mixed_op_events);
    }

    for (int support_id = 0; support_id < depths_[OpType::SUPPORT]; support_id++) {
      for (int cpu_id = 0; cpu_id < depths_[OpType::CPU]; cpu_id++) {
        // Get sequential index and prepare space all CPU Ops
        auto queue_idxs = QueueIdxs{support_id, cpu_id, 0, 0};
        PlaceWorkspace<OpType::CPU>(cpu_workspaces_, queue_idxs, graph, tensor_to_store_queue,
                                    mixed_op_stream, gpu_op_stream, mixed_op_events);
      }
    }

    for (int support_id = 0; support_id < depths_[OpType::SUPPORT]; support_id++) {
      for (int cpu_id = 0; cpu_id < depths_[OpType::CPU]; cpu_id++) {
        for (int mixed_id = 0; mixed_id < depths_[OpType::MIXED]; mixed_id++) {
          // Get sequential index and prepare space all MIXED Ops
          auto queue_idxs = QueueIdxs{support_id, cpu_id, mixed_id, 0};
          PlaceWorkspace<OpType::MIXED>(mixed_workspaces_, queue_idxs, graph, tensor_to_store_queue,
                                        mixed_op_stream, gpu_op_stream, mixed_op_events);
        }
      }
    }

    // we reuse the loop for Mixed with GPU as they are in sync
    for (int support_id = 0; support_id < depths_[OpType::SUPPORT]; support_id++) {
      for (int cpu_id = 0; cpu_id < depths_[OpType::CPU]; cpu_id++) {
        for (int mixed_id = 0; mixed_id < depths_[OpType::MIXED]; mixed_id++) {
          // Get sequential index and prepare space all GPU Ops
          auto queue_idxs = QueueIdxs{support_id, cpu_id, mixed_id, mixed_id};
          PlaceWorkspace<OpType::GPU, OpType::MIXED>(gpu_workspaces_, queue_idxs, graph,
                                                     tensor_to_store_queue, mixed_op_stream,
                                                     gpu_op_stream, mixed_op_events);
        }
      }
    }
  }

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx);

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
    DALI_ENFORCE(node.op_type == op_type,
                 "Wrong variant of method selected. OpType does not match.");
    return GetWorkspace<op_type>(idxs, graph, node.partition_index);
  }

 private:
  QueueSizes queue_sizes_;
  QueueIdxs depths_;
  // ws_id -> op_id -> workspace
  std::vector<std::vector<SupportWorkspace>> support_workspaces_;
  std::vector<std::vector<HostWorkspace>> cpu_workspaces_;
  std::vector<std::vector<MixedWorkspace>> mixed_workspaces_;
  std::vector<std::vector<DeviceWorkspace>> gpu_workspaces_;

  template <OpType op_type, OpType group_as = op_type, typename T>
  void PlaceWorkspace(T &workspaces, QueueIdxs idxs, const OpGraph &graph,
                      const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                      cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
                      const std::vector<std::vector<cudaEvent_t>> &mixed_op_events) {
    int sequential_ws_idx = SequentialIndex(idxs, depths_, group_as);
    workspaces[sequential_ws_idx].resize(graph.NumOp(op_type));
    for (OpPartitionId partition_idx = 0; partition_idx < graph.NumOp(op_type); partition_idx++) {
      auto &node = graph.Node(OpType::GPU, partition_idx);
      workspaces[sequential_ws_idx][partition_idx] =
          CreateWorkspace<op_type>(graph, node, tensor_to_store_queue, mixed_op_stream,
                                   gpu_op_stream, mixed_op_events, idxs);
    }
  }
};

template <>
inline AOT2_WS_Policy::ws_t<OpType::SUPPORT> AOT2_WS_Policy::GetWorkspace<OpType::SUPPORT>(
    QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
  int sequential_ws_idx = SequentialIndex(idxs, depths_, OpType::SUPPORT);
  return support_workspaces_[sequential_ws_idx][partition_idx];
}

template <>
inline AOT2_WS_Policy::ws_t<OpType::CPU> AOT2_WS_Policy::GetWorkspace<OpType::CPU>(
    QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
  int sequential_ws_idx = SequentialIndex(idxs, depths_, OpType::CPU);
  return cpu_workspaces_[sequential_ws_idx][partition_idx];
}

template <>
inline AOT2_WS_Policy::ws_t<OpType::MIXED> AOT2_WS_Policy::GetWorkspace<OpType::MIXED>(
    QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
  int sequential_ws_idx = SequentialIndex(idxs, depths_, OpType::MIXED);
  return mixed_workspaces_[sequential_ws_idx][partition_idx];
}

template <>
inline AOT2_WS_Policy::ws_t<OpType::GPU> AOT2_WS_Policy::GetWorkspace<OpType::GPU>(
    QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
  // TODO(klecki) due to handling GPU with mixed we have to call here index for Mixed
  int sequential_ws_idx = SequentialIndex(idxs, depths_, OpType::MIXED);
  return gpu_workspaces_[sequential_ws_idx][partition_idx];
}



/**
 * @brief Ahead Of Time Workspace Policy
 *
 * Creates all required workspaces during InitializeWorkspaceStore,
 * and provides references to the as required.
 * Inteded to be used with UniforQueuePolicy.
 */
struct AOT_WS_Policy {
  template <OpType op_type>
  using ws_t = op_type_to_workspace_t<op_type> &;

  void InitializeWorkspaceStore(const OpGraph &graph,
                                const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                                cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
                                const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                                const QueueSizes idxs) {
    DALI_ENFORCE(idxs.cpu_size == idxs.gpu_size, "This policy does not support splited queues");
    queue_size_ = idxs.cpu_size;
    wss_.resize(queue_size_);
    for (int i = 0; i < queue_size_; i++) {
      PrepareWSB(i, graph, tensor_to_store_queue, mixed_op_stream, gpu_op_stream, mixed_op_events);
    }
  }

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph,
                                                OpPartitionId partition_idx) {
    auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
    return ws_vec[partition_idx];
  }

  template <OpType op_type>
  ws_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
    DALI_ENFORCE(node.op_type == op_type,
                 "Wrong variant of method selected. OpType does not match.");
    auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
    return ws_vec[node.partition_index];
  }

 private:
  void PrepareWSB(int queue_idx, const OpGraph &graph,
                  const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                  cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
                  const std::vector<std::vector<cudaEvent_t>> &mixed_op_events) {
    // Clear any old data setup
    wss_[queue_idx].Clear();
    wss_[queue_idx].Resize(graph.NumOp(OpType::SUPPORT), graph.NumOp(OpType::CPU),
                           graph.NumOp(OpType::MIXED), graph.NumOp(OpType::GPU));

    for (int i = 0; i < graph.NumOp(); i++) {
      auto &node = graph.Node(i);
      VALUE_SWITCH(node.op_type, op_type_static,
          (OpType::SUPPORT, OpType::CPU, OpType::MIXED, OpType::GPU),
      (
        auto &ws = GetWorkspace<op_type_static>(QueueIdxs{queue_idx}, graph, node);
        ws = CreateWorkspace<op_type_static>(graph, node, tensor_to_store_queue, mixed_op_stream,
           gpu_op_stream, mixed_op_events, QueueIdxs{queue_idx});
      ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
    }
  }
  struct WorkspaceBlob {
    workspace_store_t op_data;

    void Resize(int support, int cpu, int mixed, int gpu) {
      std::get<static_cast<int>(OpType::SUPPORT)>(op_data).resize(support);
      std::get<static_cast<int>(OpType::CPU)>(op_data).resize(cpu);
      std::get<static_cast<int>(OpType::MIXED)>(op_data).resize(mixed);
      std::get<static_cast<int>(OpType::GPU)>(op_data).resize(gpu);
    }

    void Clear() {
      std::get<static_cast<int>(OpType::SUPPORT)>(op_data).clear();
      std::get<static_cast<int>(OpType::CPU)>(op_data).clear();
      std::get<static_cast<int>(OpType::MIXED)>(op_data).clear();
      std::get<static_cast<int>(OpType::GPU)>(op_data).clear();
    }
  };
  vector<WorkspaceBlob> wss_;
  int queue_size_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_WORKSPACE_POLICY_H_
