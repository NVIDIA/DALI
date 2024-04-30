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

#ifndef DALI_PIPELINE_EXECUTOR2_DYNAMIC_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_DYNAMIC_EXEC_GRAPH_H_

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <variant>

#include "../graph.h"
#include "workspace_cache.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

#include "dali/core/exec/tasking.h"

namespace dali {
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

class ExecNode {
 public:
  ExecNode() = default;
  explicit ExecNode(OperatorBase *op) : op(op) {}
  explicit ExecNode(PipelineOutputTag) : is_pipeline_output(true) {}

  std::vector<const ExecEdge *> inputs, outputs;

  std::shared_ptr<tasking::Semaphore> concurrency;
  std::shared_ptr<tasking::Semaphore> output_queue_limit;

  OperatorBase *op = nullptr;
  bool is_pipeline_output = false;

  tasking::SharedTask prev, main_task, release_outputs;

  CachedWorkspace GetWorkspace(const WorkspaceParams &params) {
    auto ws = workspace_cache_.Get(params);
    if (!ws) {
      if (op) {
        ws = CreateOpWorkspace();
      } else {
        ws = CreateOutputWorkspace();
      }
    }
    ApplyWorkspaceParams(*ws, params);
    return ws;
  }

  CachedWorkspace CreateOutputWorkspace();
  CachedWorkspace CreateOpWorkspace();

  void PutWorkspace(CachedWorkspace ws);

  WorkspaceCache workspace_cache_;

  void NextIter() {
    prev = std::move(main_task);
    release_outputs.reset();
  }

  void CreateMainTask(std::shared_ptr<Iteration> iter, const WorkspaceParams &params);
  void AddDataDeps();
  void CreateAuxTasks();
  std::optional<tasking::TaskFuture> Launch(tasking::Scheduler &sched);
};

struct ExecGraph {
  std::list<ExecNode> nodes;
  std::list<ExecEdge> edges;

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
    if (consumer)
      consumer->inputs.push_back(&edge);
  }
};

class Iteration {
 public:
  int64_t id = 0;
  tasking::TaskFuture result;
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_DYNAMIC_EXEC_GRAPH_H_
