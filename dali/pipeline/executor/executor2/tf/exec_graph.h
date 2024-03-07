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

#include "third_party/taskflow/taskflow/taskflow.hpp"  // TODO(michalz): Add it to cmake

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
};

inline void ApplyWorkspaceParams(Workspace &ws, const WorkspaceParams &params) {
  ws.SetThreadPool(params.thread_pool);
  ws.set_output_order(params.order);
}

inline WorkspaceParams GetWorkspaceParams(const Workspace &ws) {
  WorkspaceParams params;
  params.thread_pool = ws.HasThreadPool() ? &ws.GetThreadPool() : nullptr;
  params.order = ws.output_order();
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

  bool IsEssential() const {
    for (auto *edge : outputs) {
      if (edge->consumer == nullptr)  // pipeline output
        return true;
    }
    if (op->GetSpec().GetSchema().IsNoPrune())
      return true;
    return false;
  }

  mutable bool visited = false;
};

struct ExecGraph {
  std::list<ExecNode> nodes;
  std::list<ExecEdge> edges;

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

  /**
   * @brief Runs a depth-first search to topologiclaly sort and prune the graph
   */
  void sort_and_prune(std::vector<ExecNode *> &sorted) {
    for (auto &n : nodes)
      n.visited = false;

    sorted.clear();
    sorted.reserve(nodes.size());

    for (auto &node : nodes) {
      if (node.IsEssential())
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

using SchedEdge = DataEdge<SchedNode>;

struct SchedGraph;

class DLL_PUBLIC SchedNode {
 public:
  ExecNode *definition;
  span<const SchedEdge> inputs, outputs;

  std::unique_ptr<Workspace> ws;

  tf::Task main_task, release_output;
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

  SchedGraph &operator=(const SchedGraph &g) {
    nodes.resize(g.nodes.size());
    edges = g.edges;

    int V = nodes.size();
    int E = edges.size();

    auto edge_fixup = edges.data() - g.edges.data();
    auto node_fixup = nodes.data() - g.nodes.data();

    for (auto &e : edges) {
      if (e.producer) {
        e.producer += node_fixup;
        assert(e.producer >= nodes.data() && e.producer < nodes.data() + nodes.size());
      }
      if (e.consumer) {
        e.consumer += node_fixup;
        assert(e.consumer >= nodes.data() && e.consumer < nodes.data() + nodes.size());
      }
    }

    for (int i = 0; i < V; i++) {
      nodes[i].definition = g.nodes[i].definition;
      nodes[i].inputs = nodes[i].outputs = {};
      nodes[i].ws = nodes[i].definition->GetWorkspace(GetWorkspaceParams(*g.nodes[i].ws));

      if (!g.nodes[i].inputs.empty())
        nodes[i].inputs =
            span(g.nodes[i].inputs.data() + edge_fixup, g.nodes[i].inputs.size());
      if (!g.nodes[i].outputs.empty())
        nodes[i].outputs =
            span(g.nodes[i].outputs.data() + edge_fixup, g.nodes[i].outputs.size());

      assert(nodes[i].inputs.data() == nullptr ||
             (nodes[i].inputs.data() >= edges.data() &&
              nodes[i].inputs.data() + nodes[i].inputs.size() <= edges.data() + edges.size()));

      assert(nodes[i].outputs.data() == nullptr ||
             (nodes[i].outputs.data() >= edges.data() &&
              nodes[i].outputs.data() + nodes[i].outputs.size() <= edges.data() + edges.size()));
    }

    return *this;
  }

  void init(ExecGraph &def, const WorkspaceParams &params) {
    std::unordered_map<ExecNode *, int> node_indices(def.nodes.size());
    nodes.clear();
    edges.clear();

    std::vector<ExecNode *> sorted;
    def.sort_and_prune(sorted);

    int num_edges = 0;
    for (auto *node : sorted) {
      node_indices.insert({node, node_indices.size()});
      num_edges += node->inputs.size() + node->outputs.size();
    }

    edges.resize(num_edges);
    nodes.resize(sorted.size());

    int i = 0;
    int e = 0;
    for (auto &exec_node : sorted) {
      auto &sched_node = nodes[i++];
      sched_node.definition = exec_node;
      sched_node.ws = sched_node.definition->GetWorkspace(params);
      SchedEdge *inp = &edges[e];
      for (auto *exec_edge : exec_node->inputs) {
        assert(e < (int)edges.size());
        auto &sched_edge = edges[e++];
        sched_edge.producer_output_idx = exec_edge->producer_output_idx;
        sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
        if (exec_edge->producer)
          sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
        if (exec_edge->consumer)
          sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
      }
      SchedEdge *out = &edges[e];
      for (auto *exec_edge : exec_node->outputs) {
        assert(e < (int)edges.size());
        auto &sched_edge = edges[e++];
        sched_edge.producer_output_idx = exec_edge->producer_output_idx;
        sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
        if (exec_edge->producer)
          sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
        if (exec_edge->consumer)
          sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
      }
      SchedEdge *end = &edges[e];
      sched_node.inputs = span(inp, out);
      sched_node.outputs = span(out, end);
    }
    assert(static_cast<size_t>(e) == edges.size());
  }

  static auto from_def(ExecGraph &def, const WorkspaceParams &params) {
    return std::make_shared<SchedGraph>(def, params);
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
      node.release_output.reset();
    }
  }

  auto clone() const {
    return std::make_shared<SchedGraph>(*this);
  }
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_EXEC_TF_GRAPH_H_
