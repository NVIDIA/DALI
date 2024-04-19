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

#ifndef DALI_PIPELINE_EXECUTOR2_TF_EXEC_GRAPH_H_
#define DALI_PIPELINE_EXECUTOR2_TF_EXEC_GRAPH_H_

#include <cassert>
#include <functional>
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

  tf::Semaphore concurrency{1};
  std::optional<tf::Semaphore> output_queue;

  OperatorBase *op = nullptr;
  bool essential = false;

  std::queue<std::unique_ptr<Workspace>> workspaces;

  std::unique_ptr<Workspace> GetWorkspace(const WorkspaceParams &params) {
    std::unique_ptr<Workspace> ret;
    if (!workspaces.empty()) {
      ret = std::move(workspaces.front());
      workspaces.pop();
    } else {
      ret = CreateWorkspace();
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

  void PutWorkspace(std::unique_ptr<Workspace> ws) {
    workspaces.push(std::move(ws));
  }

  mutable bool visited = false;
};

struct ExecGraph {
  std::list<ExecNode> nodes;
  std::list<ExecEdge> edges;

  std::vector<ExecEdge *> inputs, outputs;

  template <typename... Args>
  ExecNode *add_node(Args &&...args) {
    return &nodes.emplace_back(std::forward<Args>(args)...);
  }

  void link(ExecNode *producer, int out_idx, ExecNode *consumer, int in_idx) {
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


  void mark_essential_nodes() {
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
  void sort_and_prune(std::vector<ExecNode *> &sorted) {
    for (auto &n : nodes)
      n.visited = false;

    mark_essential_nodes();

    sorted.clear();
    sorted.reserve(nodes.size());

    for (auto &node : nodes) {
      if (node.essential)
        sort_subgraph(sorted, &node);
    }
  }

  template <typename NodePtrs>
  void sort_subgraph(NodePtrs &sorted, ExecNode *node) {
    if (node->visited)
      return;
    node->visited = true;
    for (auto e : node->inputs)
      sort_subgraph(sorted, e->producer);

    sorted.push_back(node);
  };

};


class SchedNode;

struct SchedEdge : public DataEdge<SchedNode> {
  bool pipeline_output = false;
};

struct SchedGraph;

class DLL_PUBLIC SchedNode {
 public:
  ExecNode *definition;
  span<const SchedEdge> inputs, outputs;

  std::unique_ptr<Workspace> ws;

  tf::Task main_task, release_outputs;
  // TODO(michalz): sync with GPU ops - we only need it when the following op needs to access the
  // results on host; GPU-side dependencies can be scheduled on the stream without a need to engage
  // the CPU part of the executor.
  // tf::Semaphore gpu_dependency;

  void schedule(std::shared_ptr<SchedGraph> eg, tf::Taskflow &flow);

  /// Runs operator's `Setup` and resizes the outputs
  void task_setup();
  /// Runs the operator's `Run` method and
  void task_run();
  void task_reset_inputs();
  void task_propagate_outputs();
  void task_reset_outputs();
};


struct DLL_PUBLIC SchedGraph : public std::enable_shared_from_this<SchedGraph> {
  SchedGraph() = default;
  SchedGraph(SchedGraph &&other) = default;
  SchedGraph(const SchedGraph &other) {
    *this = other;
  }

  explicit SchedGraph(ExecGraph &def, const WorkspaceParams &params) {
    init(def, params);
  }

  std::vector<SchedNode> nodes;
  std::vector<SchedEdge> edges;
  std::vector<SchedEdge *> outputs;

  SchedGraph &operator=(const SchedGraph &g);

  void init(ExecGraph &exec, const WorkspaceParams &params);

  static auto from_exec(ExecGraph &exec, const WorkspaceParams &params) {
    return std::make_shared<SchedGraph>(exec, params);
  }


  void schedule(tf::Taskflow &flow) {
    clear_tasks();
    for (auto &node : nodes)
      schedule_node(flow, node);
  }

  void schedule_node(tf::Taskflow &flow, SchedNode &node) {
    if (!node.main_task.empty())
      return;  // already scheduled
    for (auto &in : node.inputs) {
      assert(!in.producer->main_task.empty() && "Graph not sorted");
    }
    node.schedule(shared_from_this(), flow);
  }

  void clear_tasks() {
    for (auto &node : nodes) {
      node.main_task.reset();
      node.release_outputs.reset();
    }
  }

  auto clone() const {
    return std::make_shared<SchedGraph>(*this);
  }
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXEC_TF_GRAPH_H_
