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
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/util/stream_pool.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

/**
 * @brief Basic executor for dali graphs. This executor enables
 * prefetching of results by maintaining two copies of output
 * buffers, so that we can produce data into one while the
 * other is in use by the user.
 */
class DLL_PUBLIC Executor {
 public:
  using ExecutorCallback = std::function<void(void)>;

  DLL_PUBLIC inline Executor(int batch_size, int num_thread, int device_id,
                             size_t bytes_per_sample_hint, bool set_affinity = false,
                             int max_num_stream = -1, int prefetch_queue_depth = 2)
      : batch_size_(batch_size),
        device_id_(device_id),
        bytes_per_sample_hint_(bytes_per_sample_hint),
        queue_depth_(prefetch_queue_depth),
        stream_pool_(max_num_stream, true),
        event_pool_(max_num_stream),
        thread_pool_(num_thread, device_id, set_affinity),
        exec_error_(false),
        cb_(nullptr) {
    DALI_ENFORCE(batch_size_ > 0, "Batch size must be greater than 0.");
    DALI_ENFORCE(device_id >= 0, "Device id must be non-negative.");
  }

  DLL_PUBLIC virtual ~Executor() = default;

  DLL_PUBLIC virtual void Build(OpGraph *graph, vector<string> output_names);

  DLL_PUBLIC virtual void Init() {}

  DLL_PUBLIC virtual void RunCPU();

  DLL_PUBLIC virtual void RunMixed();

  DLL_PUBLIC virtual void RunGPU();

  DLL_PUBLIC virtual void Outputs(DeviceWorkspace *ws);

  DLL_PUBLIC virtual void ShareOutputs(DeviceWorkspace *ws);

  DLL_PUBLIC virtual void ReleaseOutputs();

  DLL_PUBLIC virtual void SetCompletionCallback(ExecutorCallback cb);

  friend class ExecutorTest;

  DISABLE_COPY_MOVE_ASSIGN(Executor);

 protected:
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
  vector<WorkspaceBlob> wss_;

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

  void PruneUnusedGraphNodes();

  virtual std::vector<int> GetTensorQueueSizes(const OpGraph &graph);

  void SetupWorkspacesForGraph(int queue_idx);

  virtual void SetupOutputInfo(const OpGraph &graph);

  std::vector<int> GetMemoryHints(const OpNode &node);

  void PrepinData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                  const OpGraph &graph);

  void PresizeData(std::vector<tensor_data_store_queue_t> &tensor_to_store_queue,
                   const OpGraph &graph);

  void SetupOutputQueuesForGraph();

  template <DALIOpType op_type>
  workspace_t<op_type> &GetWorkspace(int queue_idx, OpPartitionId partition_idx);

  template <DALIOpType op_type>
  workspace_t<op_type> &GetWorkspace(int queue_idx, const OpNode &node);

  template <DALIOpType op_type>
  void SetupInputOutput(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
                        const QueueIdxs idxs);

  // template <DALIOpType op_type>
  // void SetupPinned(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
  //                  const QueueIdxs idxs);

  template <DALIOpType op_type>
  void SetupStreamsAndEvents(workspace_t<op_type> &ws, const OpGraph &graph, const OpNode &node,
                             const QueueIdxs idxs);

  template <DALIOpType op_type>
  workspace_t<op_type> CreateWorkspace(const OpGraph &graph, const OpNode &node,
                                       const QueueIdxs idxs);

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
  int queue_depth_;
  int previous_gpu_queue_idx_ = -1;

  vector<string> output_names_;

  // Meta-data about our stage outputs for fast lookup
  std::vector<TensorNodeId> pipeline_outputs_;
  std::vector<EventList> gpu_output_events_;

  // Buffers are rotated between being 'free', where the
  // pipeline is ok to fill them with data, 'ready', where
  // they are already full of prepared data, and 'in-use',
  // where the user currently owns that buffer. A buffer
  // is marked as in-use when it is returned as and output.
  // The buffer is then returned the the ready queue the
  // next time Ouputs() is called.
  std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  std::mutex ready_mutex_, free_mutex_;
  std::condition_variable ready_cond_, free_cond_;

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
  std::queue<int> mixed_work_queue_, gpu_work_queue_;
  std::mutex mixed_mutex_, gpu_mutex_;

  OpGraph *graph_ = nullptr;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
  std::vector<std::string> errors_;
  std::mutex errors_mutex_;
  bool exec_error_;
  ExecutorCallback cb_;
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  // MixedOpId -> queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED to GPU Ops
  std::vector<std::vector<cudaEvent_t>> mixed_op_events_;
};

#define USE_EXECUTOR_MEMBERS()            \
 protected:                               \
  using Executor::WorkspaceBlob;          \
  using Executor::wss_;                   \
  using Executor::batch_size_;            \
  using Executor::device_id_;             \
  using Executor::bytes_per_sample_hint_; \
  using Executor::queue_depth_;           \
  using Executor::output_names_;          \
  using Executor::gpu_output_events_;     \
  using Executor::ready_queue_;           \
  using Executor::ready_mutex_;           \
  using Executor::ready_cond_;            \
  using Executor::graph_;                 \
  using Executor::stream_pool_;           \
  using Executor::event_pool_;            \
  using Executor::thread_pool_

template <DALIOpType op_type>
workspace_t<op_type> &Executor::GetWorkspace(int queue_idx, OpPartitionId partition_idx) {
  auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[queue_idx].op_data);
  return ws_vec[partition_idx];
}

template <DALIOpType op_type>
workspace_t<op_type> &Executor::GetWorkspace(int queue_idx, const OpNode &node) {
  DALI_ENFORCE(node.op_type == op_type,
               "Wrong variant of method selected. DALIOpType does not match.");
  auto &ws_vec = std::get<static_cast<size_t>(op_type)>(wss_[queue_idx].op_data);
  return ws_vec[node.partition_index];
}

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
void Executor::SetupInputOutput(workspace_t<op_type> &ws, const OpGraph &graph,
                                const OpNode &node, const QueueIdxs idxs) {
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
                                                            tensor_to_store_queue_[tid],
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
        get_queue<DALIOpType::SUPPORT, DALITensorDevice::CPU>(tensor_to_store_queue_[tid]);
    auto tensor = queue[idxs[DALIOpType::MIXED]];  // TODO(klecki): check queueueueueuing
    ws.AddArgumentInput(tensor, arg_pair.first);
  }

  for (int j = 0; j < node.spec.NumOutput(); ++j) {
    auto tid = node.children_tensors[j];
    auto tensor_device = graph.Tensor(tid).producer_edge.storage_device;
    VALUE_SWITCH(tensor_device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
    (
      add_output<op_type, device_static>(ws, tensor_to_store_queue_[tid], idxs[op_type]);
    ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
  }
}

// template <DALIOpType op_type>
// void Executor::SetupPinned(workspace_t<op_type> &, const OpGraph &, const OpNode &,
//                            const QueueIdxs) {
//   /* No-op if we are not MIXED MakeContigous node */
// }

// // TODO(klecki): this should be handled on Tensor level?
// template <>
// inline void Executor::SetupPinned<DALIOpType::MIXED>(MixedWorkspace &ws, const OpGraph &graph,
//                                                      const OpNode &node, const QueueIdxs idxs) {
//   for (int j = 0; j < node.spec.NumRegularInput(); ++j) {
//     auto tid = node.parent_tensors[j];
//     // Use pinned memory only when it is useful
//     if (node.spec.name() == "MakeContiguous" && node.spec.NumOutput() == 1 &&
//         node.spec.OutputDevice(0) == "gpu") {
//       auto &parent_tensor_queue =
//           get_queue<DALIOpType::CPU, DALITensorDevice::CPU>(tensor_to_store_queue_[tid]);
//       auto &tensor = parent_tensor_queue[idxs[DALIOpType::MIXED]];
//       // SetPinned(tensor, true);
//     }
//   }
// }

template <DALIOpType op_type>
void Executor::SetupStreamsAndEvents(workspace_t<op_type> &ws, const OpGraph &graph,
                                     const OpNode &node, const QueueIdxs idxs) {
  /* No-op if we are not Mixed or GPU */
}

template <>
inline void Executor::SetupStreamsAndEvents<DALIOpType::MIXED>(MixedWorkspace &ws,
                                                               const OpGraph &graph,
                                                               const OpNode &node,
                                                               const QueueIdxs idxs) {
  // We assign unique stream to mixed ops.
  // This ensures that we won't have false dependencies
  // between mixed ops and the previous iterations
  // gpu ops.
  ws.set_stream(mixed_op_stream_);
  ws.set_event(mixed_op_events_[node.partition_index][idxs[DALIOpType::MIXED]]);
}

template <>
inline void Executor::SetupStreamsAndEvents<DALIOpType::GPU>(DeviceWorkspace &ws,
                                                             const OpGraph &graph,
                                                             const OpNode &node,
                                                             const QueueIdxs idxs) {
  // I/O pipeline is always going to be launched alongside
  // some other GPU work (like DL training).
  // Therefore it is not necessary to use more than
  // 1 stream for GPU ops, even though we may not fill
  // the whole GPU with just I/O pipeline kernels
  // by doing so.
  ws.set_stream(gpu_op_stream_);
  for (const auto &p : node.parents) {
    if (graph.NodeType(p) == DALIOpType::MIXED) {
      const auto &parent_op = graph.Node(p);
      // We need to block on this op's event to
      // make sure that we respect the dependency
      ws.AddParentEvent(mixed_op_events_[parent_op.partition_index][idxs[DALIOpType::MIXED]]);
    }
  }
}

template <DALIOpType op_type>
workspace_t<op_type> Executor::CreateWorkspace(const OpGraph &graph, const OpNode &node,
                                               const QueueIdxs idxs) {
  workspace_t<op_type> ws;
  SetupInputOutput<op_type>(ws, graph, node, idxs);
  // SetupPinned<op_type>(ws, graph, node, idxs);
  SetupStreamsAndEvents<op_type>(ws, graph, node, idxs);
  return ws;
}

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
