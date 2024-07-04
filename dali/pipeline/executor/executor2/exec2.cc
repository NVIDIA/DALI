// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <queue>
#include <unordered_map>
#include <utility>
#include "dali/pipeline/executor/executor2/exec2.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/stream_assignment.h"
#include "dali/core/cuda_stream_pool.h"


namespace dali {
namespace exec2 {

class Executor2::Impl {
 public:
  explicit Impl(const Config &config) : config_(config) {
  }

  void Build(const graph::OpGraph &graph) {
    DeviceGuard dg(config_.device.value_or(CPU_ONLY_DEVICE_ID));
    graph_.Lower(graph);
    AnalyzeGraph();
    CheckNodeTypes();
    CalculatePrefetchDepth();
    SetupStreams();
    InitThreadPool();
    InitExecutor();
  }

  void Run() {
    if (!exec_)
      throw std::runtime_error("The executor is not initialized.");
    InitIteration();
    graph_.Launch(*exec_);
  }

  void Prefetch() {
    if (!exec_)
      throw std::runtime_error("The executor is not initialized.");
    for (int i = 0; i < prefetch_depth_; i++) {
      Run();
    }
  }

  Workspace PopOutputs() {
    if (pending_outputs_.empty())
      throw std::out_of_range("All pending outputs were already popped.");
    auto fut = std::move(pending_outputs_.front());
    pending_outputs_.pop();
    return fut.Value<Workspace>();
  }

  void InitIteration() {
    WorkspaceParams params{};
    params.batch_size = InferBatchSize();
    graph_.PrepareIteration(InitIterationData(), params);
  }

 private:
  std::shared_ptr<IterationData> InitIterationData() {
    auto iter_data = std::make_shared<IterationData>();
    iter_data->iteration_index = iter_index_++;
    return iter_data;
  }


  int InferBatchSize() {
    assert(!"Not implemented!");
  }

  void AnalyzeGraph() {
    CountNodes();
    // FindInputNodes();
  }

  void CountNodes() {
    for (auto &n : graph_.nodes) {
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
    for (auto &node : graph_.nodes) {
      if (node.inputs.empty() && node.def) {
        int op_depth;
        if (node.def->spec.TryGetArgument(depth, "queue_depth"))
          depth = std::max(depth, op_depth);
      }
    }
    prefetch_depth_ = depth;
  }

  void InitThreadPool() {
    if (graph_info_.num_cpu > 0) {
      tp_ = std::make_unique<ThreadPool>(
        config_.thread_pool_threads,
        config_.device.value_or(CPU_ONLY_DEVICE_ID),
        config_.set_affinity,
        "Executorv_v2");
    } else {
      tp_.reset();
    }
  }

  void InitExecutor() {
    exec_ = std::make_unique<tasking::Executor>(config_.operator_threads);
    exec_->Start();
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
    for (auto &node : graph_.nodes) {
      auto stream_idx = assignment[&node];

      node.env.order = stream_idx.has_value()
                     ? AccessOrder(streams_[*stream_idx])
                     : AccessOrder::host();
    }
  }

  Config config_;
  int prefetch_depth_ = 1;

  struct GraphInfo {
    int num_cpu = 0;
    int num_mixed = 0;
    int num_gpu = 0;
    int num_cpu_roots = 0;
    int num_mixed_roots = 0;
    int num_gpu_roots = 0;

    std::vector<ExecNode *> input_nodes;
  } graph_info_;

  std::unique_ptr<ThreadPool> tp_;
  std::queue<tasking::TaskFuture> pending_outputs_;
  std::vector<CUDAStreamLease> streams_;

  ExecGraph graph_;
  std::unique_ptr<tasking::Executor> exec_;

  int iter_index_ = 0;
};


///////////////////////////////
// Executor2

Executor2::Executor2(const Config &config) : impl_(std::make_unique<Impl>(config)) {
}

Executor2::~Executor2() = default;

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
}

ExecutorMetaMap Executor2::GetExecutorMeta() {
    return {};
}

void Executor2::Shutdown() {
}

Checkpoint &Executor2::GetCurrentCheckpoint() {
}

void Executor2::RestoreStateFromCheckpoint(const Checkpoint &cpt) {
}

int Executor2::InputFeedCount(std::string_view input_name) {
}

OperatorBase *Executor2::GetOperator(std::string_view name) {
}


}  // namespace exec2
}  // namespace dali
