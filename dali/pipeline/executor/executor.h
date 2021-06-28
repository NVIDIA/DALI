// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/executor/queue_metadata.h"
#include "dali/pipeline/executor/queue_policy.h"
#include "dali/pipeline/executor/workspace_policy.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/graph/op_graph_verifier.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/util/event_pool.h"
#include "dali/pipeline/util/stream_pool.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/workspace_data_factory.h"

namespace dali {

struct DLL_PUBLIC ExecutorMeta {
  size_t real_size;
  size_t max_real_size;
  size_t reserved;
  size_t max_reserved;
};
using ExecutorMetaMap = std::unordered_map<std::string, std::vector<ExecutorMeta>>;

namespace detail {
// This is stream callback used on GPU stream to indicate that GPU work for this
// pipeline run is finished
static void gpu_finished_callback(cudaStream_t stream, cudaError_t status, void *userData);

// helper function to concatenate ExecutorMetaMap maps
static void AppendToMap(ExecutorMetaMap &ret, ExecutorMetaMap &in_stats, std::mutex &mutex);

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
  DLL_PUBLIC virtual void EnableMemoryStats(bool enable_memory_stats = false) = 0;
  DLL_PUBLIC virtual ExecutorMetaMap GetExecutorMeta() = 0;

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
  DLL_PUBLIC inline Executor(int max_batch_size, int num_thread, int device_id,
                             size_t bytes_per_sample_hint, bool set_affinity = false,
                             int max_num_stream = -1, int default_cuda_stream_priority = 0,
                             QueueSizes prefetch_queue_depth = QueueSizes{2, 2})
      : max_batch_size_(max_batch_size),
        device_id_(device_id),
        bytes_per_sample_hint_(bytes_per_sample_hint),
        callback_(nullptr),
        stream_pool_(max_num_stream, true, default_cuda_stream_priority),
        event_pool_(),
        thread_pool_(num_thread, device_id, set_affinity),
        exec_error_(false),
        queue_sizes_(prefetch_queue_depth),
        mixed_op_stream_(0),
        gpu_op_stream_(0),
        enable_memory_stats_(false) {
    DALI_ENFORCE(max_batch_size_ > 0, "Max batch size must be greater than 0.");

    stage_queue_depths_ = QueuePolicy::GetQueueSizes(prefetch_queue_depth);
  }

  DLL_PUBLIC void EnableMemoryStats(bool enable_memory_stats = false) override {
    enable_memory_stats_ = enable_memory_stats;
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
  DLL_PUBLIC ExecutorMetaMap GetExecutorMeta() override;

  DLL_PUBLIC void ShutdownQueue() {
    QueuePolicy::SignalStop();
  }

  DISABLE_COPY_MOVE_ASSIGN(Executor);

 protected:
  DLL_PUBLIC void RunCPUImpl();
  DLL_PUBLIC void RunMixedImpl();
  DLL_PUBLIC void RunGPUImpl();

  template<typename T>
  inline void GetMaxSizesCont(T &in, size_t &max_out_size, size_t &max_reserved_size) {
    auto out_size = in.nbytes();
    auto reserved_size = in.capacity();
    max_out_size = std::max<size_t>(std::ceil((out_size * 1.0) / in.ntensor()), max_out_size);
    max_reserved_size = std::max<size_t>(std::ceil((reserved_size * 1.0) / in.ntensor()),
                                         max_reserved_size);
  }

  template<typename T>
  inline void GetMaxSizesNonCont(T &in, size_t &max_out_size, size_t &max_reserved_size) {
    for (size_t j = 0; j < in.ntensor(); ++j) {
      max_out_size = std::max(in[j].nbytes(), max_out_size);
      max_reserved_size = std::max(in[j].capacity(), max_reserved_size);
    }
  }

  template<typename backend>
  inline void GetMaxSizes(TensorList<backend> &in, size_t &max_out_size,
                          size_t &max_reserved_size) {
    GetMaxSizesCont(in, max_out_size, max_reserved_size);
  }

  template<typename backend>
  inline void GetMaxSizes(TensorVector<backend> &in, size_t &max_out_size,
                          size_t &max_reserved_size) {
    if (in.IsContiguous()) {
      GetMaxSizesCont(in, max_out_size, max_reserved_size);
    } else {
      GetMaxSizesNonCont(in, max_out_size, max_reserved_size);
    }
  }

  template <typename W>
  inline void FillStats(ExecutorMetaMap &memory_stats, W &ws, std::string op_name,
                        std::mutex &write_mutex) {
    if (enable_memory_stats_) {
        size_t out_size = 0;
        size_t max_out_size = 0;
        size_t reserved_size = 0;
        size_t max_reserved_size = 0;
        std::lock_guard<std::mutex> lck(write_mutex);
        auto &stats = memory_stats[op_name];
        stats.resize(ws.NumOutput(), {0, 0});

        for (int i = 0; i < ws.NumOutput(); ++i) {
          out_size = 0;
          max_out_size = 0;
          reserved_size = 0;
          max_reserved_size = 0;
          if (ws.template OutputIsType<CPUBackend>(i)) {
            auto &out = ws.template OutputRef<CPUBackend>(i);
            out_size = out.nbytes();
            reserved_size = out.capacity();
            GetMaxSizes(out, max_out_size, max_reserved_size);
          } else {
            auto &out = ws.template OutputRef<GPUBackend>(i);
            out_size = out.nbytes();
            reserved_size = out.capacity();
            GetMaxSizes(out, max_out_size, max_reserved_size);
          }
          stats[i].real_size = std::max(out_size, stats[i].real_size);
          stats[i].max_real_size = std::max(max_out_size, stats[i].max_real_size);
          stats[i].reserved = std::max(reserved_size, stats[i].reserved);
          stats[i].max_reserved = std::max(max_reserved_size, stats[i].max_reserved);
        }
      }
  }

  void HandleError(const std::string &stage, const OpNode &op_node, const std::string &message) {
    // handle internal Operator names that start with underscore
    const auto &op_name =
        op_node.spec.name()[0] == '_' ? op_node.spec.name().substr(1) : op_node.spec.name();

    bool need_instance_name = false;
    for (int op_id = 0; op_id < graph_->NumOp(); op_id++) {
      if (op_id != op_node.id && graph_->Node(op_id).spec.name() == op_node.spec.name()) {
        need_instance_name = true;
        break;
      }
    }
    if (need_instance_name) {
      HandleError(make_string("Error when executing ", stage, " operator ", op_name,
                              ", instance name: \"", op_node.instance_name, "\", encountered:\n",
                              message));
    } else {
      HandleError(make_string("Error when executing ", stage, " operator ", op_name,
                              " encountered:\n", message));
    }
  }

  void HandleError(const std::string& message = "Unknown exception") {
    {
      std::lock_guard<std::mutex> errors_lock(errors_mutex_);
      errors_.push_back(message);
    }
    exec_error_ = true;
    ShutdownQueue();
  }

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
    inline EventList() = default;
    inline EventList(int size, EventPool *event_pool) {
      DALI_ENFORCE(event_pool != nullptr);
      for (int i = 0; i < size; ++i) {
        events_.push_back(event_pool->GetEvent());
      }
    }

    inline cudaEvent_t GetEvent(int idx) { return events_[idx]; }

    inline bool empty() const {
      return events_.empty();
    }

   private:
    vector<cudaEvent_t> events_;
  };

  int max_batch_size_, device_id_;
  size_t bytes_per_sample_hint_;
  cudaEvent_t mixed_stage_event_ = {};
  cudaEvent_t gpu_stage_event_ = {};

  vector<string> output_names_;

  // Meta-data about our stage outputs for fast lookup
  std::vector<TensorNodeId> pipeline_outputs_;

  // If there are GPU outputs from given stages, we have to wait for them to finish.
  // Those EventList will contain the number of events matching the size of prefetch queue
  // for given stage only if there are GPU events. Otherwise they should be empty,
  // so we can skip recording and waiting for synchronous CPU buffers.
  EventList mixed_output_events_, gpu_output_events_;

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

  std::queue<int> batch_sizes_cpu_, batch_sizes_mixed_, batch_sizes_gpu_;

  OpGraph *graph_ = nullptr;
  // we need to keep this above the stream_pool_ so we still have it when the stream_pool_
  // destructor runs and it waits for streams to finish
  ExecutorCallback callback_;
  StreamPool stream_pool_;
  EventPool event_pool_;
  ThreadPool thread_pool_;
  std::vector<std::string> errors_;
  mutable std::mutex errors_mutex_;
  bool exec_error_;
  QueueSizes queue_sizes_;
  std::vector<tensor_data_store_queue_t> tensor_to_store_queue_;
  cudaStream_t mixed_op_stream_, gpu_op_stream_;
  // MixedOpId -> queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED to GPU Ops
  MixedOpEventMap mixed_op_events_;
  // queue_idx -> cudaEvent_t
  // To introduce dependency from MIXED stage to GPU stage for callback only
  // in some edge cases where there are no operators
  std::vector<cudaEvent_t> mixed_callback_events_;

  std::atomic<bool> enable_memory_stats_;
  ExecutorMetaMap cpu_memory_stats_, mixed_memory_stats_, gpu_memory_stats_;
  std::mutex cpu_memory_stats_mutex_;
  std::mutex mixed_memory_stats_mutex_;
  std::mutex gpu_memory_stats_mutex_;

  /// Graph nodes, which define batch size for the entire graph
  std::vector<BatchSizeProvider *> batch_size_providers_;

 private:
  template <typename InputRef>
  static bool SetDefaultLayoutIfNeeded(InputRef &in, const OpSchema &schema, int in_idx) {
    if (!in.GetLayout().empty()) return false;
    auto default_layout = schema.GetInputLayout(in_idx, in.shape().sample_dim(), in.GetLayout());
    if (default_layout.empty()) return false;
    in.SetLayout(default_layout);
    return true;
  }

  template <typename Workspace>
  void RunHelper(OpNode &op_node, Workspace &ws);

  void RethrowError() const {
    std::lock_guard<std::mutex> errors_lock(errors_mutex_);
    // TODO(klecki): collect all errors
    std::string message = errors_.empty()
                          ? QueuePolicy::IsStopSignaled() && !exec_error_
                            ? "Stop signaled"
                            : "Unknown error"
                          : errors_.front();

    // TODO(michalz): rethrow actual error through std::exception_ptr instead of
    //                converting everything to runtime_error
    throw std::runtime_error(message);
  }

  void DiscoverBatchSizeProviders() {
    for (Index i = 0; i < graph_->NumOp(); i++) {
      auto bsp = dynamic_cast<BatchSizeProvider *>(graph_->Node(i).op.get());
      if (!bsp) continue;
      batch_size_providers_.emplace_back(bsp);
    }
  }

  int InferBatchSize(const std::vector<BatchSizeProvider *> &batch_size_providers) const;

  void PreRun();
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
ExecutorMetaMap Executor<WorkspacePolicy, QueuePolicy>::GetExecutorMeta() {
  ExecutorMetaMap ret;
  detail::AppendToMap(ret, cpu_memory_stats_, cpu_memory_stats_mutex_);
  detail::AppendToMap(ret, mixed_memory_stats_, mixed_memory_stats_mutex_);
  detail::AppendToMap(ret, gpu_memory_stats_, gpu_memory_stats_mutex_);
  return ret;
}

template <typename WorkspacePolicy, typename QueuePolicy>
void Executor<WorkspacePolicy, QueuePolicy>::Build(OpGraph *graph, vector<string> output_names) {
  DALI_ENFORCE(graph != nullptr, "Input graph is nullptr.");
  DALI_ENFORCE(graph->NumOp() > 0, "Graph has no operators.");
  graph->InstantiateOperators();  // ..if not done already

  output_names_ = std::move(output_names);
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
  tensor_to_store_queue_ =
      CreateBackingStorageForTensorNodes(*graph_, max_batch_size_, queue_sizes);
  // Setup stream and events that will be used for execution
  if (device_id_ != CPU_ONLY_DEVICE_ID) {
    DeviceGuard g(device_id_);
    mixed_op_stream_ = stream_pool_.GetStream();
    gpu_op_stream_ = stream_pool_.GetStream();
    mixed_op_events_ =
        CreateEventsForMixedOps(event_pool_, *graph_, stage_queue_depths_[OpType::MIXED]);

    // Create events used to synchronize stages using gpu with themselves
    mixed_stage_event_ = event_pool_.GetEvent();
    gpu_stage_event_ = event_pool_.GetEvent();
  }

  PrepinData(tensor_to_store_queue_, *graph_);

  // Presize the workspaces based on the hint
  PresizeData(tensor_to_store_queue_, *graph_);

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  // For each set of outputs, setup another set of
  // workspaces so that nothing has to be altered
  // during execution (this is necessary for
  // asynchronous executors that can overlap work issue)
  WorkspacePolicy::InitializeWorkspaceStore(*graph_, tensor_to_store_queue_, &thread_pool_,
                                            mixed_op_stream_, gpu_op_stream_, mixed_op_events_,
                                            queue_sizes_);

  // Producer-consumer queues info
  SetupOutputQueuesForGraph();

  DiscoverBatchSizeProviders();
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
    VALUE_SWITCH(storage_dev, storage_dev_static, (StorageDevice::GPU, StorageDevice::CPU),
    (
      VALUE_SWITCH(op_type, op_type_static, (OpType::CPU, OpType::MIXED, OpType::GPU),
      (
        auto &queue = get_queue<op_type_static, storage_dev_static>(
            tensor_to_store_queue_[out_tensor_id]);
        auto stage_output_idx = output_idx[op_type_static];
        ws->AddOutput(PresentAsTensorList(queue[stage_output_idx]));
      ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
    ), DALI_FAIL("Invalid storage device"));  // NOLINT(whitespace/parens)
  }
  // We than need to wait for GPU outputs from Mixed & GPU stages that are computed asynchronously.
  // If the output event list is not empty, it means that there are outputs on GPU that we
  // have to wait for.
  if (!mixed_output_events_.empty()) {
    auto queue_idx = output_idx[OpType::MIXED];
    CUDA_CALL(cudaEventSynchronize(mixed_output_events_.GetEvent(queue_idx)));
  }
  if (!gpu_output_events_.empty()) {
    auto queue_idx = output_idx[OpType::GPU];
    CUDA_CALL(cudaEventSynchronize(gpu_output_events_.GetEvent(queue_idx)));
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
      // Do not prune the node if it has a preserve flag
      if (!node.op->CanBePruned()) continue;

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

  // If there are GPU outputs from given stages, we have to wait for them
  auto has_gpu_output = [] (OpType stage_type, const auto &pipeline_outputs,
                            const OpGraph &graph_to_check) {
    for (auto tid : pipeline_outputs) {
      const auto &tensor = graph_to_check.Tensor(tid);
      const auto &producer_node = graph_to_check.Node(tensor.producer.node);
      if (producer_node.op_type == stage_type) {
        if (tensor.producer.storage_device == StorageDevice::GPU) {
          return true;
        }
      }
    }
    return false;
  };

  if (has_gpu_output(OpType::MIXED, pipeline_outputs_, graph)) {
    mixed_output_events_ = EventList(stage_queue_depths_[OpType::MIXED], &event_pool_);
  }
  if (has_gpu_output(OpType::GPU, pipeline_outputs_, graph)) {
    gpu_output_events_ = EventList(stage_queue_depths_[OpType::GPU], &event_pool_);
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
std::vector<int> Executor<WorkspacePolicy, QueuePolicy>::GetTensorQueueSizes(const OpGraph &graph) {
  std::vector<int> result;
  // By default we need one vector
  result.resize(graph.NumTensor(), 1);
  auto output_ids = graph.GetOutputs(output_names_, true);
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
      if (node.spec.name() == "MakeContiguous" && node.spec.NumOutput() == 1) {
        auto &parent_tensor_queue =
            get_queue<OpType::CPU, StorageDevice::CPU>(tensor_to_store_queue_[tid]);
        for (auto &tensor : parent_tensor_queue) {
          tensor->set_pinned(node.spec.OutputDevice(0) == "gpu");
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
  DomainTimeRange tr("[DALI][Executor] PresizeData");

  auto should_reserve = [](auto &storage, Index hint, StorageDevice dev) -> bool {
    if (dev == StorageDevice::CPU) {
      return hint && storage->is_pinned();
    }
    return hint;
  };

  auto reserve_batch = [](auto &storage, const OperatorBase &op, Index hint, int batch_size) {
    // If the Op Can Infer Outputs we want to do one contiguous pre-allocation
    if (op.CanInferOutputs()) {
      storage->reserve(hint * batch_size);
    } else {
      storage->reserve(hint, batch_size);
    }
  };

  // To avoid handling the arguments several times for each operator that
  // has more than one output, we go over the operators instead of tensors
  for (int i = 0; i < graph.NumOp(); i++) {
    auto &node = graph.Node(i);
    auto hints = GetMemoryHints(node);
    VALUE_SWITCH(node.op_type, op_type_static,
        (OpType::CPU, OpType::MIXED, OpType::GPU),
    (
      // For all tensors we produce
      for (size_t j = 0; j < node.children_tensors.size(); j++) {
        auto &tensor = graph.Tensor(node.children_tensors[j]);
        Index hint = hints[j];
        VALUE_SWITCH(tensor.producer.storage_device, dev_static,
            (StorageDevice::CPU, StorageDevice::GPU),
        (
          auto& queue = get_queue<op_type_static, dev_static>(tensor_to_store_queue[tensor.id]);
          for (auto storage : queue) {
            if (should_reserve(storage, hint, dev_static)) {
              reserve_batch(storage, *node.op, hint, max_batch_size_);
            }
            if (node.op->CanInferOutputs()) {
              storage->SetContiguous(true);
            }
          }
        ), DALI_FAIL("Invalid StorageDevice"));  // NOLINT(whitespace/parens)
      }
    ), DALI_FAIL("Invalid op type"));  // NOLINT(whitespace/parens)
  }
}

template <typename WorkspacePolicy, typename QueuePolicy>
std::vector<int> Executor<WorkspacePolicy, QueuePolicy>::GetMemoryHints(const OpNode &node) {
  std::vector<int> hints;
  GetSingleOrRepeatedArg(node.spec, hints, "bytes_per_sample_hint", node.spec.NumOutput());
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

void AppendToMap(ExecutorMetaMap &ret, ExecutorMetaMap &in_stats, std::mutex &mutex) {
  const std::lock_guard<std::mutex> lock(mutex);
  ret.insert(in_stats.begin(), in_stats.end());
}

}  // namespace detail

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
