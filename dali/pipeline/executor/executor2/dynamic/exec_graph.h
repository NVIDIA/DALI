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

#ifndef DALI_PIPELINE_EXECUTOR2_EXEC_DYNAMIC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_EXEC_DYNAMIC_GRAPH_H_

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <variant>

#include "../graph.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

#include "dali/core/exec/tasking.h"

namespace dali {
namespace exec2 {

class SchedNode;
class ExecNode;

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

struct WorkspaceParams {
  ThreadPool  *thread_pool = nullptr;
  AccessOrder  order = AccessOrder::host();
  std::optional<int> batch_size = 0;  // TODO(michalz): add more batch size logic
};

inline void ApplyWorkspaceParams(Workspace &ws, const WorkspaceParams &params) {
  ws.SetThreadPool(params.thread_pool);
  ws.set_output_order(params.order);
  if (params.batch_size.has_value())
    ws.SetBatchSizes(*params.batch_size);
}

inline WorkspaceParams GetWorkspaceParams(const Workspace &ws) {
  WorkspaceParams params = {};
  params.thread_pool = ws.HasThreadPool() ? &ws.GetThreadPool() : nullptr;
  params.order = ws.output_order();
  if (ws.NumOutput())
    params.batch_size = ws.GetRequestedBatchSize(0);
  else if (ws.NumInput())
    params.batch_size = ws.GetInputBatchSize(0);
  return params;
}

class ExecNode {
 public:
  ExecNode() = default;
  explicit ExecNode(OperatorBase *op) : op(op) {

  }

  std::vector<const ExecEdge *> inputs, outputs;

  std::shared_ptr<tasking::Semaphore> concurrency;
  std::shared_ptr<tasking::Semaphore> output_queue_limit;

  OperatorBase *op = nullptr;
  bool essential = false;

  std::mutex workspace_lock;
  std::queue<std::unique_ptr<Workspace>> workspaces;

  tasking::SharedTask prev, main_task, release_outputs;

  std::unique_ptr<Workspace> GetWorkspace(const WorkspaceParams &params) {
    std::unique_ptr<Workspace> ret;
    {
      std::unique_lock g(workspace_lock);
      if (!workspaces.empty()) {
        ret = std::move(workspaces.front());
        workspaces.pop();
      } else {
        g.unlock();
        ret = CreateWorkspace();
      }
    }
    ApplyWorkspaceParams(*ret, params);
    return ret;
  }

  std::unique_ptr<Workspace> CreateWorkspace() const {
    auto &spec = op->GetSpec();
    auto ws = std::make_unique<Workspace>();
    for (int i = 0; i < spec.NumInput(); i++) {
      bool arg = spec.IsArgumentInput(i);
      bool gpu = spec.InputDevice(i) == "gpu";
      assert((inputs[i]->device == StorageDevice::GPU) == gpu);
      if (arg) {
        ws->AddArgumentInput(spec.ArgumentInputName(i), nullptr);
      } else if (gpu) {
        ws->AddInput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
      } else {
        ws->AddInput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
      }
    }
    for (int i = 0; i < spec.NumOutput(); i++) {
      bool gpu = spec.OutputDevice(i) == "gpu";
      assert((outputs[i]->device == StorageDevice::GPU) == gpu);
      if (gpu) {
        ws->AddOutput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
      } else {
        ws->AddOutput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
      }
    }
    return ws;
  }

  void PutWorkspace(std::unique_ptr<Workspace> ws);

  void NextIter() {
    prev = std::move(main_task);
    release_outputs.reset();
  }

  void CreateMainTask(const WorkspaceParams &params);
  void AddDataDeps();
  void CreateAuxTasks();
  void LaunchSilent(tasking::Scheduler &sched);
  tasking::TaskFuture Launch(tasking::Scheduler &sched);

  mutable bool visited = false;
};

class OpTaskFunc {
 private:
  OpTaskFunc(ExecNode *node, std::unique_ptr<Workspace> ws)
  : node_(node), ws_(std::move(ws)) {}

  auto GetTaskRunnable() && {
    return [self = std::move(*this)](tasking::Task *t) mutable {
      self.task_ = t;
      return self.Run();
    };
  }

 public:
  OpTaskFunc(OpTaskFunc &&) = default;
  OpTaskFunc(const OpTaskFunc &) {
    std::cerr << "This constructor is here only because std::function requires "
                 "the functor to be copy-constructible. We never actually copy the target.\n"
                 "See C++23 std::move_only_function." << std::endl;
    std::abort();
  }

  static tasking::SharedTask CreateTask(ExecNode *node, std::unique_ptr<Workspace> ws) {
    return tasking::Task::Create(
      ws->NumOutput(),
      OpTaskFunc(node, std::move(ws)).GetTaskRunnable());
  }

 private:
  using OpTaskOutputs = SmallVector<std::any, 8>;

  OpTaskOutputs Run();

  tasking::Task *task_ = nullptr;
  ExecNode *node_ = nullptr;
  std::unique_ptr<Workspace> ws_;

  template <typename Backend>
  const auto &TaskInput(int i) const {
    return task_->GetInputValue<const std::shared_ptr<TensorList<Backend>> &>(i);
  }

  void SetWorkspaceInputs();
  void SetupOp();
  void RunOp();
  void ResetWorkspaceInputs();
  OpTaskOutputs GetWorkspaceOutputs();
};

struct ExecGraph {
  std::list<ExecNode> nodes;
  std::list<ExecEdge> edges;

  std::vector<ExecEdge *> inputs, outputs;

  template <typename... Args>
  ExecNode *AddNode(Args &&...args) {
    return &nodes.emplace_back(std::forward<Args>(args)...);
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


  void MarkEssentialNodes() {
    for (auto &node : nodes)
      node.essential = false;

    for (auto *e : outputs) {
      if (e->producer)
        e->producer->essential = true;
    }

    for (auto &node : nodes) {
      if (!node.essential) {
        auto *op = node.op;
        if (op && op->GetSpec().GetSchema().IsNoPrune())
          node.essential = true;
      }
    }
  }


  /**
   * @brief Runs a depth-first search to topologiclaly sort and prune the graph
   */
  void SortAndPrune(std::vector<ExecNode *> &sorted) {
    for (auto &n : nodes)
      n.visited = false;

    MarkEssentialNodes();

    sorted.clear();
    sorted.reserve(nodes.size());

    for (auto &node : nodes) {
      if (node.essential)
        SortSubgraph(sorted, &node);
    }
  }

  template <typename NodePtrs>
  void SortSubgraph(NodePtrs &sorted, ExecNode *node) {
    if (node->visited)
      return;
    node->visited = true;
    for (auto e : node->inputs)
      SortSubgraph(sorted, e->producer);

    sorted.push_back(node);
  };

  std::vector<ExecNode *> essential;
};

class Iteration {
 public:


  tasking::TaskFuture outputs;
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXEC_DYNAMIC_GRAPH_H_
