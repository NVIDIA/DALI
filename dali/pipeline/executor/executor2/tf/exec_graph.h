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
  ThreadPool  *tp;
  AccessOrder  order;
};

class ExecNode {
 public:
  ExecNode() = default;
  explicit ExecNode(std::function<void(Workspace &)> fn) : task_fn(std::move(fn)) {}

  std::vector<const ExecEdge *> inputs, outputs;
  std::function<void(Workspace &)> task_fn;

  tf::Semaphore concurrency{1};
  std::optional<tf::Semaphore> output_queue;

  OperatorBase *op = nullptr;

  std::queue<std::unique_ptr<Workspace>> workspaces;

  std::unique_ptr<Workspace> GetWorkspace() {
    if (!workspaces.empty()) {
      auto ret = std::move(workspaces.front());
      workspaces.pop();
      return ret;
    } else {
      return CreateWorkspace();
    }
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

  void link(ExecNode *producer, int out_idx, ExecNode *consumer, int in_idx) {
    auto &edge = edges.emplace_back();
    edge.producer = producer;
    edge.producer_output_idx = out_idx;
    edge.consumer = consumer;
    edge.consumer_input_idx = in_idx;

    producer->outputs.push_back(&edge);
    consumer->inputs.push_back(&edge);
  }

  void sort(std::vector<ExecNode *> sorted) {
    for (auto &n : nodes)
      n.visited = false;

    sorted.clear();
    sorted.reserve(nodes.size());

    for (auto &node : nodes)
      sort_subgraph(sorted, &node);
  }

  template <typename NodePtrs>
  void sort_subgraph(NodePtrs &sorted, ExecNode *node) {
    if (node->visited)
      return;
    for (auto e : node->inputs)
      sort_subgraph(sorted, e->producer);

    sorted.push_back(node);
  };

};


class SchedNode;

using SchedEdge = DataEdge<SchedNode>;

struct SchedGraph;

class SchedNode {
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
};


struct SchedGraph : public std::enable_shared_from_this<SchedGraph> {
  SchedGraph() = default;
  SchedGraph(SchedGraph &&other) = default;
  SchedGraph(const SchedGraph &other) {
    *this = other;
  }

  explicit SchedGraph(ExecGraph &def) {
    init(def);
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
      nodes[i].ws = nodes[i].definition->GetWorkspace();

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

  void init(ExecGraph &def) {
    std::map<ExecNode *, int> node_indices;
    nodes.clear();
    edges.clear();

    std::vector<ExecNode *> sorted;
    def.sort(sorted);

    int num_edges = 0;
    for (auto &node : def.nodes) {
      node_indices.insert({&node, node_indices.size()});
      num_edges += node.inputs.size() + node.outputs.size();
    }

    edges.resize(num_edges);
    nodes.resize(def.nodes.size());

    int i = 0;
    int e = 0;
    for (auto &exec_node : sorted) {
      auto &sched_node = nodes[i++];
      sched_node.definition = exec_node;
      sched_node.ws = sched_node.definition->GetWorkspace();
      SchedEdge *inp = &edges[e];
      for (auto *exec_edge : exec_node->inputs) {
        assert(e < (int)edges.size());
        auto &sched_edge = edges[e++];
        sched_edge.producer_output_idx = exec_edge->producer_output_idx;
        sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
        sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
        sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
      }
      SchedEdge *out = &edges[e];
      for (auto *exec_edge : exec_node->outputs) {
        assert(e < (int)edges.size());
        auto &sched_edge = edges[e++];
        sched_edge.producer_output_idx = exec_edge->producer_output_idx;
        sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
        sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
        sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
      }
      SchedEdge *end = &edges[e];
      sched_node.inputs = span(inp, out);
      sched_node.outputs = span(out, end);
    }
    assert(static_cast<size_t>(e) == edges.size());
  }

  static auto from_def(ExecGraph &def) {
    return std::make_shared<SchedGraph>(def);
  }


  void schedule(tf::Taskflow &flow) {
    clear_tasks();
    for (auto &node : nodes)
      schedule_node(flow, node);
  }

  void schedule_node(tf::Taskflow &flow, SchedNode &node) {
    if (!node.main_task.empty())
      return;  // already scheduled
    for (auto &in : node.inputs)
      schedule_node(flow, *in.producer);
    assert(node.main_task.empty() && "Cycle detected");
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
