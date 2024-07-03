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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

#include "dali/pipeline/executor/executor2/workspace_cache.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

#include "dali/core/exec/tasking.h"

namespace dali {
namespace graph {
class OpGraph;
struct OpNode;
}  // namespace graph
namespace exec2 {

class ExecNode;
class Iteration;

template <typename NodeType = ExecNode>
struct DataEdge {
  NodeType *producer = nullptr;
  NodeType *consumer = nullptr;
  int producer_output_idx = 0;
  int consumer_input_idx = 0;
  StorageDevice device = {};

  constexpr bool operator==(const DataEdge &other) const {
    return producer == other.producer &&
           consumer == other.consumer &&
           producer_output_idx == other.producer_output_idx &&
           consumer_input_idx == other.consumer_input_idx &&
           device == other.device;
  }

  constexpr bool operator!=(const DataEdge &other) const {
    return !(*this == other);
  }
};

using ExecEdge = DataEdge<ExecNode>;

struct PipelineOutputTag {};

class DLL_PUBLIC ExecNode {
 public:
  ExecNode() = default;
  ExecNode(std::unique_ptr<OperatorBase> op, const graph::OpNode *def = nullptr);
  explicit ExecNode(PipelineOutputTag) : is_pipeline_output(true) {}

  std::vector<ExecEdge *> inputs, outputs;

  std::shared_ptr<tasking::Semaphore> concurrency;
  std::shared_ptr<tasking::Semaphore> output_queue_limit;

  std::unique_ptr<OperatorBase> op;

  tasking::SharedTask prev, main_task, release_outputs;

  ExecEnv env = {};

  void PutWorkspace(CachedWorkspace ws);

  CachedWorkspace GetWorkspace(std::shared_ptr<IterationData> iter_data,
                               WorkspaceParams params) {
    auto ws = workspace_cache_.Get(params);
    if (!ws) {
      if (op) {
        ws = CreateOpWorkspace();
      } else {
        ws = CreateOutputWorkspace();
      }
    }
    if (!params.env)
      params.env = &env;
    ApplyWorkspaceParams(*ws, params);
    ws->InjectIterationData(iter_data);
    return ws;
  }

  const graph::OpNode *def = nullptr;
  OpType device = OpType::CPU;
  bool is_pipeline_output = false;

  mutable bool visited = false;

 private:
  CachedWorkspace CreateOutputWorkspace();
  CachedWorkspace CreateOpWorkspace();


  WorkspaceCache workspace_cache_;
  CachedWorkspace current_workspace_;

  void NextIter() {
    prev = std::move(main_task);
    release_outputs.reset();
  }

  friend class ExecGraph;

  void CreateMainTask(std::shared_ptr<IterationData> iter, const WorkspaceParams &params);
  void AddDataDeps();
  void CreateAuxTasks();
  std::optional<tasking::TaskFuture> Launch(tasking::Scheduler &sched);
};

class DLL_PUBLIC ExecGraph {
 public:
  std::list<ExecNode> nodes;
  std::list<ExecEdge> edges;
  std::unordered_map<const graph::OpNode *, ExecNode *> def2exec;

  template <typename... Args>
  ExecNode *AddNode(Args &&...args) {
    return &nodes.emplace_back(std::forward<Args>(args)...);
  }

  ExecNode *AddOutputNode() {
    return &nodes.emplace_back(PipelineOutputTag());
  }

  void Link(ExecNode *producer, int out_idx, ExecNode *consumer, int in_idx) {
    auto &edge = edges.emplace_back();
    edge.producer = producer;
    edge.producer_output_idx = out_idx;
    edge.consumer = consumer;
    edge.consumer_input_idx = in_idx;

    if (producer)
      producer->outputs.push_back(&edge);
    if (consumer) {
      consumer->inputs.resize(std::max<size_t>(consumer->inputs.size(), in_idx + 1));
      consumer->inputs[in_idx] = &edge;
    }
  }

  void PrepareIteration(const std::shared_ptr<IterationData> &iter_data,
                        const WorkspaceParams &params);

  tasking::TaskFuture Launch(tasking::Scheduler &sched);

  void Lower(const graph::OpGraph &def);
};

class Iteration {
 public:
  int64_t id = 0;
  tasking::TaskFuture result;
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_GRAPH_H_

