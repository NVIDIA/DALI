// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <map>
#include <queue>
#include <unordered_map>
#include <utility>
#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/executor/executor2/exec2.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/stream_assignment.h"

namespace dali {
namespace exec2 {

namespace {

void LimitBackendConcurrency(ExecGraph &graph, OpType backend, int max_concurrency = 1) {
  auto sem = std::make_shared<tasking::Semaphore>(max_concurrency);
  for (auto &n : graph.Nodes()) {
    if (n.backend == backend)
        n.concurrency = sem;
  }
  graph.Invalidate();
}

void ApplyConcurrencyLimit(ExecGraph &graph, OperatorConcurrency concurrency) {
  switch (concurrency) {
    case OperatorConcurrency::Full:
      // TODO(michalz): Fix ThreadPool.
      LimitBackendConcurrency(graph, OpType::CPU);
      break;  // other operators have no restrictions
    case OperatorConcurrency::Backend:
      LimitBackendConcurrency(graph, OpType::CPU);
      LimitBackendConcurrency(graph, OpType::GPU);
      LimitBackendConcurrency(graph, OpType::MIXED);
      break;
    case OperatorConcurrency::None:
      {
        auto sem = std::make_shared<tasking::Semaphore>(1);
        for (auto &n : graph.Nodes())
            n.concurrency = sem;
      }
      break;
    default:
      assert(!"Unexpected concurrency policy value.");
      break;
  }
}

}  // namespace

class Executor2::Impl {
 public:
  explicit Impl(const Config &config) : config_(config) {
  }
  ~Impl() {
    Shutdown();
  }

  enum class State {
    New = 0,
    Building,
    Built,
    Running,
    ShutdownRequested,
    ShutDown,
  };

  void Build(const graph::OpGraph &graph) {
    if (state_ != State::New)
      throw std::logic_error("Already built.");
    state_ = State::Building;
    DeviceGuard dg(config_.device.value_or(CPU_ONLY_DEVICE_ID));
    graph_.Lower(graph);
    BuildNodeDict();
    AnalyzeGraph();
    CheckNodeTypes();
    CalculatePrefetchDepth();
    ApplyConcurrencyLimit(graph_, config_.concurrency);
    SetupStreams();
    SetupThreadPool();

    last_iter_data_ = InitIterationData(-1);
    if (last_iter_data_->checkpoint)
      PopulateInitialCheckpoint(*last_iter_data_->checkpoint);

    state_ = State::Built;
    Start();
  }

  void Run() {
    if (state_ != State::Running)
      throw std::runtime_error("The executor is not initialized.");
    InitIteration();
    pending_outputs_.push(graph_.Launch(*exec_));
  }

  void Prefetch() {
    for (int i = 0; i < prefetch_depth_; i++) {
      Run();
    }
  }

  Workspace PopOutputs() {
    if (pending_outputs_.empty())
      throw std::out_of_range("All pending outputs were already popped.");
    auto fut = std::move(pending_outputs_.front());
    pending_outputs_.pop();
    auto &pipe_out = fut.Value<const PipelineOutput &>();
    auto ws = pipe_out.workspace;
    last_iter_data_ = ws.GetIterationData();
    if (ws.has_event())
      CUDA_CALL(cudaEventSynchronize(ws.event()));
    ws.set_event(nullptr);
    return ws;
  }

  void InitIteration() {
    WorkspaceParams params{};
    params.batch_size = config_.max_batch_size;
    graph_.PrepareIteration(InitIterationData(iter_index_++), params);
  }

  int InputFeedCount(std::string_view) {
    return prefetch_depth_;
  }

  OperatorBase *GetOperator(std::string_view input_name) const {
    auto it = node_map_.find(input_name);
    if (it == node_map_.end())
      return nullptr;
    return it->second->op.get();
  }

  const SharedIterData &LastIterData() const {
    return last_iter_data_;
  }

  void Shutdown() {
    if (state_ != State::Running)
      return;
    state_ = State::ShutdownRequested;
    if (exec_)
      exec_->Shutdown();
    state_ = State::ShutDown;
  }

  void EnableCheckpointing(bool enabled) {
    config_.checkpointing = enabled;
  }

  bool CheckpointingEnabled() const {
    return config_.checkpointing;
  }

 private:
  State state_ = State::New;

  std::shared_ptr<IterationData> InitIterationData(int iter_index) {
    auto iter_data = std::make_shared<IterationData>();
    iter_data->iteration_index = iter_index;
    if (config_.checkpointing) {
      iter_data->checkpoint = CreateCheckpoint(iter_data->iteration_index);
    }
    return iter_data;
  }

  std::shared_ptr<Checkpoint> CreateCheckpoint(int64_t iteration_index) {
    auto cpt = std::make_shared<Checkpoint>();
    cpt->SetIterationId(iter_index_ + 1);
    for (auto &n : graph_.Nodes()) {
      if (!n.instance_name.empty())
        cpt->AddOperator(n.instance_name);
    }
    return cpt;
  }

  void PopulateInitialCheckpoint(Checkpoint &cpt) {
    for (auto &n : graph_.Nodes()) {
      n.op->SaveState(cpt.GetOpCheckpoint(n.instance_name), n.env.order);
    }
  }

  void BuildNodeDict() {
    for (auto &n : graph_.Nodes())
      if (!n.instance_name.empty())
        node_map_[n.instance_name] = &n;
  }

  void AnalyzeGraph() {
    CountNodes();
  }

  void CountNodes() {
    for (auto &n : graph_.Nodes()) {
      switch (NodeType(&n)) {
      case OpType::CPU:
        graph_info_.num_cpu++;
        if (n.inputs.empty())
          graph_info_.num_cpu_roots++;
        break;
      case OpType::GPU:
        graph_info_.num_gpu++;
        if (n.inputs.empty())
          graph_info_.num_gpu_roots++;
        break;
      case OpType::MIXED:
        graph_info_.num_mixed++;
        if (n.inputs.empty())
          graph_info_.num_mixed_roots++;
        break;
      default:
        break;
      }
    }
  }

  void CheckNodeTypes() {
    if (graph_info_.num_gpu + graph_info_.num_mixed > 0 && !config_.device.has_value())
      throw std::invalid_argument("The graph contains nodes that require a GPU but the config "
                                  "doesn't specify a device id.");
  }

  void CalculatePrefetchDepth() {
    int depth = 1;
    if (graph_info_.num_cpu_roots > 0)
      depth = std::max(depth, config_.cpu_queue_depth);
    if (graph_info_.num_mixed_roots + graph_info_.num_gpu_roots > 0)
      depth = std::max(depth, config_.gpu_queue_depth);
    for (auto &node : graph_.Nodes()) {
      if (node.inputs.empty() && node.op) {
        int op_depth;
        if (node.op->GetSpec().TryGetArgument(op_depth, "queue_depth"))
          depth = std::max(depth, op_depth);
      }
    }
    prefetch_depth_ = depth;
  }

  void SetupThreadPool() {
    if (graph_info_.num_cpu > 0) {
      tp_ = std::make_unique<ThreadPool>(
        config_.thread_pool_threads,
        config_.device.value_or(CPU_ONLY_DEVICE_ID),
        config_.set_affinity,
        "Executorv_v2");
    } else {
      tp_.reset();
    }
    for (auto &n : graph_.Nodes()) {
      if (n.backend == OpType::CPU)
        n.env.thread_pool = tp_.get();
    }
  }

  void Start() {
    exec_ = std::make_unique<tasking::Executor>(config_.operator_threads);
    exec_->Start();
    state_ = State::Running;
  }

  void SetupStreams() {
    switch (config_.stream_policy) {
    case StreamPolicy::Single:
      SetupStreamsImpl<StreamPolicy::Single>();
      break;
    case StreamPolicy::PerBackend:
      SetupStreamsImpl<StreamPolicy::PerBackend>();
      break;
    case StreamPolicy::PerOperator:
      SetupStreamsImpl<StreamPolicy::PerOperator>();
      break;
    }
  }

  template <StreamPolicy policy>
  void SetupStreamsImpl() {
    StreamAssignment<policy> assignment(graph_);
    int num_streams = assignment.NumStreams();
    if (num_streams == 0)
      return;
    for (int i = 0; i < num_streams; i++)
      streams_.push_back(CUDAStreamPool::instance().Get());
    for (auto &node : graph_.Nodes()) {
      auto stream_idx = assignment[&node];

      node.env.order = stream_idx.has_value()
                     ? AccessOrder(streams_[*stream_idx])
                     : AccessOrder::host();
    }
  }

  // Configuration data

  Config config_;
  int prefetch_depth_ = 1;

  // Graph analysis

  struct GraphInfo {
    int num_cpu = 0;
    int num_mixed = 0;
    int num_gpu = 0;
    int num_cpu_roots = 0;
    int num_mixed_roots = 0;
    int num_gpu_roots = 0;
  } graph_info_;

  // Runtime environment

  std::unique_ptr<ThreadPool> tp_;
  std::queue<tasking::TaskFuture> pending_outputs_;
  std::vector<CUDAStreamLease> streams_;
  std::map<std::string, ExecNode *, std::less<>> node_map_;

  ExecGraph graph_;
  std::unique_ptr<tasking::Executor> exec_;

  // dynamic data

  int64_t iter_index_ = 0;
  SharedIterData last_iter_data_;
};


///////////////////////////////
// Executor2

Executor2::Executor2(const Config &config) : impl_(std::make_unique<Impl>(config)) {
}

Executor2::~Executor2() {
  Shutdown();
  impl_.reset();
}

void Executor2::Build(const graph::OpGraph &graph) {
  impl_->Build(graph);
}

void Executor2::Init() {
}

void Executor2::Run() {
  impl_->Run();
}

void Executor2::Prefetch() {
  impl_->Prefetch();
}


void Executor2::Outputs(Workspace *ws) {
  *ws = impl_->PopOutputs();
}

void Executor2::ShareOutputs(Workspace *ws) {
  Outputs(ws);
}

void Executor2::ReleaseOutputs() {
  // no-op
}

void Executor2::EnableMemoryStats(bool enable_memory_stats) {
}

void Executor2::EnableCheckpointing(bool checkpointing) {
  impl_->EnableCheckpointing(checkpointing);
}

ExecutorMetaMap Executor2::GetExecutorMeta() {
    return {};
}

void Executor2::Shutdown() {
  impl_->Shutdown();
}

Checkpoint &Executor2::GetCurrentCheckpoint() {
  auto iter_data = impl_->LastIterData();
  if (!iter_data) {
    throw std::runtime_error("The pipeline is not fully initialized.");
  }
  if (!iter_data->checkpoint) {
    throw std::runtime_error("The recent iteration was run without checkpoiting enabled.");
  }
  return *iter_data->checkpoint;
}

void Executor2::RestoreStateFromCheckpoint(const Checkpoint &cpt) {
  throw std::runtime_error("Not implemented");
}

int Executor2::InputFeedCount(std::string_view input_name) {
  return impl_->InputFeedCount(input_name);
}

OperatorBase *Executor2::GetOperator(std::string_view name) {
  return impl_->GetOperator(name);
}


}  // namespace exec2
}  // namespace dali
