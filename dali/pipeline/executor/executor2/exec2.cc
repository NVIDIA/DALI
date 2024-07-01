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

#include <unordered_map>
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
    SetupStreams();
    //InitThreadPool();
  }

  void Run();

  void Prefetch();

  Workspace PopOutputs();

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

 private:
  template <StreamPolicy policy>
  void SetupStreamsImpl() {
    StreamAssignment<policy> assignment(graph_);
    int num_streams = assignment.NumStreams();
    if (num_streams == 0)
      return;
    for (int i = 0; i < num_streams; i++)
      streams_.push_back(CUDAStreamPool::Get());
    for (auto &node : graph_.nodes) {
      auto stream_idx = assignment.GetStreamIdx(&node);

      node.env.order = stream_idx.has_value()
                     ? AccessOrder(streams[stream_idx])
                     : AccessOrder::host();
    }
  }

  std::unique_ptr<ThreadPool> tp_;
  ExecGraph graph_;
  Config config_;
  std::queue<tasking::TaskFuture> pending_outputs_;
  std::vector<CUDAStreamLease> streams_;
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
