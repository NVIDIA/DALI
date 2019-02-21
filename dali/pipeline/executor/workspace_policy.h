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



#include "dali/common.h"
#include "dali/error_handling.h"

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"
#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"


namespace dali {

// TODO(klecki): move to another file
struct QueueIdxs {
  int &operator[](DALIOpType op_type) {
    return idxs[static_cast<size_t>(op_type)];
  }

  const int &operator[](DALIOpType op_type) const {
    return idxs[static_cast<size_t>(op_type)];
  }

  explicit QueueIdxs(int uniform_idx)
      : idxs{uniform_idx, uniform_idx, uniform_idx, uniform_idx} {}

  private:
  std::array<int, static_cast<size_t>(DALIOpType::COUNT)> idxs = {0, 0, 0, 0};
};

struct QueueSizes {
  QueueSizes() = default;
  QueueSizes(int output_size) : cpu_size(1), mixed_size(output_size), gpu_size(output_size) {}
  QueueSizes(int cpu_size, int mixed_size, int gpu_size)
      : cpu_size(cpu_size), mixed_size(mixed_size), gpu_size(gpu_size) {}

  int cpu_size = 1, mixed_size = 1, gpu_size = 1;
};


// We instantiate the operation of adding the input only for parent op_type and device
// that are specifically allowed
// We always use queue_idx = 0 if give queue has only one element -> it is not queued
template <DALIOpType op_type, DALIOpType producer_type, DALITensorDevice device>
en_if_t<allows_op_input<op_type>(producer_type) && allows_tensor_input<op_type>(device)> add_input(
    workspace_t<op_type> &ws, const tensor_data_store_queue_t &storage, int queue_idx = 0) {
  auto &queue = get_queue<producer_type, device>(storage);
  DALI_ENFORCE(!queue.IsBuffered() || queue_idx < static_cast<int>(queue.size()),
               "Backing Tensor store queue has not enough elements.");
  auto tensor = queue[queue_idx];
  ws.AddInput(tensor);
}

// If parent op_type or device is not allowed this is a no-op
template <DALIOpType op_type, DALIOpType producer_type, DALITensorDevice device>
en_if_t<!allows_op_input<op_type>(producer_type) || !allows_tensor_input<op_type>(device)>
add_input(workspace_t<op_type>, const tensor_data_store_queue_t, int = 0) {}

// This will be only used for allowed ones (TODO(klecki) with exception of the device)
template <DALIOpType op_type, DALITensorDevice device>
void add_output(workspace_t<op_type> &ws, const tensor_data_store_queue_t &storage,
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
template <DALIOpType op_type>
void SetupInputOutput(
    workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
    const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue, const QueueIdxs idxs) {
  for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
    auto tid = node.parent_tensors[j];
    auto &parent_node = graph.Node(graph.Tensor(tid).producer_edge.node);
    auto parent_op_type = parent_node.op_type;
    auto tensor_device = graph.Tensor(tid).producer_edge.storage_device;

    VALUE_SWITCH(parent_op_type, parent_op_static,
        (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED, DALIOpType::SUPPORT),
    (
      VALUE_SWITCH(tensor_device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
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
    auto &queue =
        get_queue<DALIOpType::SUPPORT, DALITensorDevice::CPU>(tensor_to_store_queue[tid]);
    auto tensor = queue[idxs[DALIOpType::MIXED]];  // TODO(klecki): check queueueueueuing
    ws.AddArgumentInput(tensor, arg_pair.first);
  }

  for (int j = 0; j < node.spec.NumOutput(); ++j) {
    auto tid = node.children_tensors[j];
    auto tensor_device = graph.Tensor(tid).producer_edge.storage_device;
    VALUE_SWITCH(tensor_device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
    (
      add_output<op_type, device_static>(ws, tensor_to_store_queue[tid], idxs[op_type]);
    ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
  }
}

template <DALIOpType op_type>
void SetupStreamsAndEvents(workspace_t<op_type> &ws, const OpGraph &graph,
                                     const OpNode &node, cudaStream_t mixed_op_stream,
                                     cudaStream_t gpu_op_stream,
                                     const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                                     const QueueIdxs idxs) {
  /* No-op if we are not Mixed or GPU */
}

template <>
inline void SetupStreamsAndEvents<DALIOpType::MIXED>(
    MixedWorkspace &ws, const OpGraph &graph, const OpNode &node, cudaStream_t mixed_op_stream,
    cudaStream_t gpu_op_stream, const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
    const QueueIdxs idxs) {
  // We assign unique stream to mixed ops.
  // This ensures that we won't have false dependencies
  // between mixed ops and the previous iterations
  // gpu ops.
  ws.set_stream(mixed_op_stream);
  ws.set_event(mixed_op_events[node.partition_index][idxs[DALIOpType::MIXED]]);
}

template <>
inline void SetupStreamsAndEvents<DALIOpType::GPU>(
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
    if (graph.NodeType(p) == DALIOpType::MIXED) {
      const auto &parent_op = graph.Node(p);
      // We need to block on this op's event to
      // make sure that we respect the dependency
      ws.AddParentEvent(mixed_op_events[parent_op.partition_index][idxs[DALIOpType::MIXED]]);
    }
  }
}

template <DALIOpType op_type>
workspace_t<op_type> CreateWorkspace(const OpGraph &graph, const OpNode &node,
    const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue, cudaStream_t mixed_op_stream,
    cudaStream_t gpu_op_stream, const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
                                               const QueueIdxs idxs) {
  workspace_t<op_type> ws;
  SetupInputOutput<op_type>(ws, graph, node, tensor_to_store_queue, idxs);
  // SetupPinned<op_type>(ws, graph, node, idxs);
  SetupStreamsAndEvents<op_type>(ws, graph, node, mixed_op_stream, gpu_op_stream,
                                 mixed_op_events, idxs);
  return ws;
}

struct JIT_WS_Policy {
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

  template <DALIOpType op_type>
  workspace_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
    return CreateWorkspace<op_type>(graph, graph.Node(op_type, partition_idx), 
      tensor_to_store_queue_, mixed_op_stream_, gpu_op_stream_, mixed_op_events_, idxs);
  }

  template <DALIOpType op_type>
  workspace_t<op_type> GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
      DALI_ENFORCE(node.op_type == op_type,
               "Wrong variant of method selected. DALIOpType does not match.");
    return  CreateWorkspace<op_type>(graph, node, tensor_to_store_queue_, mixed_op_stream_,
        gpu_op_stream_, mixed_op_events_, idxs);
  }

 private:
  // TODO(klecki): should consider if storing copy of backing storage is good idea
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  std::vector<std::vector<cudaEvent_t>> mixed_op_events_;
  QueueSizes queue_sizes_;
};

struct AOT_WS_Policy {
  void InitializeWorkspaceStore(const OpGraph &graph,
      const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
      cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
      const std::vector<std::vector<cudaEvent_t>> &mixed_op_events,
      const QueueSizes idxs) {
    DALI_ENFORCE(idxs.cpu_size == idxs.mixed_size && idxs.mixed_size == idxs.gpu_size,
        "This policy does not support splited queues");
    queue_size_ = idxs.cpu_size;
    wss_.resize(queue_size_);
    for (int i = 0; i < queue_size_; i++) {
      PrepareWSB(i, graph, tensor_to_store_queue, mixed_op_stream, gpu_op_stream, mixed_op_events);
    }
  }

  template <DALIOpType op_type>
  workspace_t<op_type> &GetWorkspace(QueueIdxs idxs, const OpGraph &graph, OpPartitionId partition_idx) {
    auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
    return ws_vec[partition_idx];
  }

  template <DALIOpType op_type>
  workspace_t<op_type> &GetWorkspace(QueueIdxs idxs, const OpGraph &graph, const OpNode &node) {
    DALI_ENFORCE(node.op_type == op_type,
              "Wrong variant of method selected. DALIOpType does not match.");
    auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[idxs[op_type]].op_data);
    return ws_vec[node.partition_index];
  }

 private:
  void PrepareWSB(int queue_idx, const OpGraph &graph,
      const std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
      cudaStream_t mixed_op_stream, cudaStream_t gpu_op_stream,
      const std::vector<std::vector<cudaEvent_t>> &mixed_op_events) {
    // DeviceGuard g(device_id_); // TODO(klecki): really?

    // Clear any old data setup
    wss_[queue_idx].Clear();
    wss_[queue_idx].Resize(graph.NumOp(DALIOpType::SUPPORT), graph.NumOp(DALIOpType::CPU),
                graph.NumOp(DALIOpType::MIXED), graph.NumOp(DALIOpType::GPU));

    for (int i = 0; i < graph.NumOp(); i++) {
      auto &node = graph.Node(i);
      VALUE_SWITCH(node.op_type, op_type_static,
          (DALIOpType::SUPPORT, DALIOpType::CPU, DALIOpType::MIXED, DALIOpType::GPU),
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
      std::get<static_cast<int>(DALIOpType::SUPPORT)>(op_data).resize(support);
      std::get<static_cast<int>(DALIOpType::CPU)>(op_data).resize(cpu);
      std::get<static_cast<int>(DALIOpType::MIXED)>(op_data).resize(mixed);
      std::get<static_cast<int>(DALIOpType::GPU)>(op_data).resize(gpu);
    }

    void Clear() {
      std::get<0>(op_data).clear();
      std::get<1>(op_data).clear();
      std::get<2>(op_data).clear();
      std::get<3>(op_data).clear();
    }
  };
  // TODO(klecki): do not need the blob right now
  vector<WorkspaceBlob> wss_;
  int queue_size_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_WORKSPACE_POLICY_H_