// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {

class ExecGraph::Analyzer {
 public:
  void FindPinnedBuffers(ExecGraph &g) {
    // No non-cpu ops? Just mark everything as non-pinned and we're done.
    auto is_gpu_edge = [](const ExecEdge &e) { return e.device == StorageDevice::GPU; };
    bool has_gpu_buffers = std::find_if(g.edges_.begin(), g.edges_.end(), is_gpu_edge)
                           != g.edges_.end();
    if (!has_gpu_buffers) {
      for (auto &n : g.nodes_)
        for (auto &o : n.outputs)
          o.pinned = false;
      return;
    }

    // go in reverse topological order, from outputs to inputs
    for (auto it = g.nodes_.rbegin(); it != g.nodes_.rend(); ++it) {
      ExecNode &n = *it;
      if (n.is_pipeline_output)
        continue;
      SetPinnedInputs(&n);
    }
  }

  bool HasParallelConsumers(const ExecOutputDesc &out) {
    int ncons = out.consumers.size();
    // If there's just one outgoing edge from that input, we're safe.
    if (ncons <= 1)
      return false;

    // If there are multiple edges, but they point to different inputs of the same
    // consumer, then the input is effectively consumed in parallel.
    for (int i = 1; i < ncons; i++)
      if (out.consumers[i]->consumer == out.consumers[0]->consumer)
        return true;

    // Finally, let's go over all the consumers and check if they're guarded with one
    // semaphore with MaxCount() == 1. If so, then the access to the node is sequential.
    auto sem = out.consumers[0]->consumer->concurrency;
    if (!sem)
      return true;
    if (sem->MaxCount() > 1)
      return true;
    for (size_t i = 1; i < out.consumers.size(); i++)
      if (out.consumers[i]->consumer->concurrency != sem)
        return true;
    return false;
  }

  void MarkOutputsWithParallelConsumers(ExecGraph &g) {
    for (auto &n : g.nodes_) {
      for (auto &o : n.outputs)
        o.parallel_consumers = HasParallelConsumers(o);
    }
  }

 private:
  /** Sets pinnedness of the input sources
   *
   * The function goes over the inputs of the node. If the node is non-CPU, then all of its
   * CPU _regular_ inputs are marked as pinned.
   * If the node is a CPU node but passes through an input `i` directly to a pinned output `o`,
   * then the source of input `i` is also marked as pinned.
   */
  static void SetPinnedInputs(ExecNode *node) {
    assert(node->op != nullptr);
    // TODO(michalz): Update if/when we have passthrough for argument inputs
    int ninp = node->op->GetSpec().NumRegularInput();
    assert(static_cast<size_t>(ninp) <= node->inputs.size());

    if (node->backend != OpType::CPU) {
      for (int i = 0; i < ninp; i++) {
        auto *inp = node->inputs[i];
        inp->producer->outputs[inp->producer_output_idx].pinned = true;
      }
    } else if (node->op->GetSpec().GetSchema().HasPassThrough()) {
      auto &schema = node->op->GetSpec().GetSchema();
      int nout = node->outputs.size();
      for (int i = 0; i < ninp; i++) {
        auto *input = node->inputs[i];
        if (input->device != StorageDevice::CPU)  // we're not interested in non-CPU buffers
          continue;

        auto &source_output = input->producer->outputs[input->producer_output_idx];
        if (source_output.pinned)  // already pinned
          continue;

        for (int o = 0; o < nout; o++) {
          // If input `i` passes to a pinned output `o`, then the input should also be marked
          // as pinned. This will be followed in reverse topological order.
          if (node->outputs[o].pinned && schema.IsPassThrough(i, o, false)) {
            source_output.pinned = true;
            break;
          }
        }
      }
    }
  }
};

class ExecGraph::SortHelper {
 public:
  explicit SortHelper(ExecGraph &graph) : graph_(graph) {}

  using NodeList = std::list<ExecNode>;
  using NodeIt = NodeList::iterator;

  void Run() {
    sorted_.reserve(graph_.nodes_.size());

    graph::ClearVisitMarkers(graph_.nodes_);

    // Stable topological sort - if the graph is already sorted, then the order shouldn't change
    for (auto &node : graph_.nodes_)
      SortNode(&node);

    MoveBatchSizeProvidersToFront();
    RecreateNodeList();
  }

  void SortNode(ExecNode *node) {
    assert(node);
    graph::Visit visit(node);
    if (!visit)
      return;
    for (auto *edge : node->inputs) {
      assert(edge);
      SortNode(edge->producer);
    }
    int idx = sorted_.size();
    sorted_.push_back(node);
  }

  void MoveBatchSizeProvidersToFront() {
    std::stable_partition(sorted_.begin(), sorted_.end(), [](const ExecNode *n) {
      return n->is_batch_size_provider;
    });
  }

  void RecreateNodeList() {
    std::unordered_map<const ExecNode *, NodeIt> node2it;
    for (auto it = graph_.nodes_.begin(); it != graph_.nodes_.end(); ++it)
      node2it[&*it] = it;

    NodeList sorted_list;
    for (auto *node : sorted_)
      sorted_list.splice(sorted_list.end(), graph_.nodes_, node2it[node]);

    assert(graph_.nodes_.empty() && "Everything should have been moved to the sorted list");
    graph_.nodes_ = std::move(sorted_list);
  }

  ExecGraph &graph_;
  std::vector<ExecNode *> sorted_;
};

void ExecGraph::Sort() {
  if (sorted_)
    return;

  SortHelper sort(*this);
  sort.Run();
  sorted_ = true;
}

void ExecGraph::Analyze() {
  if (analyzed_)
    return;
  Analyzer a;
  a.FindPinnedBuffers(*this);
  a.MarkOutputsWithParallelConsumers(*this);
  analyzed_ = true;
}

void ExecGraph::Validate() {
  // The checks here are extremely defensive, but they're only run once.
  auto err = [](auto &&... msg) {
    throw std::logic_error(make_string("Internal error: ", msg...));
  };

  if (validated_)
    return;
  if (nodes_.empty()) {
    if (!edges_.empty())
      err("a graph without any node has edges.");
    return;
  }
  std::unordered_set<const ExecNode *> known_nodes(nodes_.size());
  std::unordered_set<const ExecEdge *> known_edges(edges_.size());

  for (auto &n : nodes_)
    known_nodes.insert(&n);
  for (auto &e : edges_) {
    known_edges.insert(&e);
  }

  for (auto &e : edges_) {
    if (!known_nodes.count(e.producer))
      err("an edge's producer is not a known node pointer.");
    if (!known_nodes.count(e.consumer))
      err("an edge's consumer is not a known node pointer.");

    if (e.producer_output_idx >= static_cast<int>(e.producer->outputs.size()))
      err("producer output index is out of range.");
    auto &consumer_edges = e.producer->outputs[e.producer_output_idx].consumers;
    if (std::count(consumer_edges.begin(), consumer_edges.end(), &e) != 1)
      err("the relevant producer's output doesn't have this edge as one of the consumers.");

    if (e.consumer->inputs[e.consumer_input_idx] != &e)
      err("inconsistent edge consumer vs consumer node's input.");
  }

  for (auto &n : nodes_) {
    if (n.op) {
      auto &spec = n.op->GetSpec();
      if (n.inputs.size() != static_cast<size_t>(spec.NumInput()))
        err("a node has a different number of inputs than used in the OpSpec");
      if (n.outputs.size() != static_cast<size_t>(spec.NumOutput()))
        err("a node has a different number of outputs than used in the OpSpec");
    }

    for (int o = 0, nout = n.outputs.size(); o < nout; o++) {
      auto &consumers = n.outputs[o].consumers;
      for (auto &e : consumers) {
        if (!known_edges.count(e))
          err("a node's output is not a known edge pointer.");
        if (e->producer != &n)
          err("a node's output's producer should always point to self.");
        if (e->producer_output_idx != o)
          err("a node's output's index must match its position in the output array.");
      }
    }
    for (int i = 0, ninp = n.inputs.size(); i < ninp; i++) {
      auto *e = n.inputs[i];
      if (!known_edges.count(e))
        err("a node's output is not a known edge pointer.");
      if (e->consumer != &n)
        err("a node's input's consumer should always point to self.");
      if (e->consumer_input_idx != i)
        err("a node's input index must match its position in the input array.");
    }

    bool is_last = &n == &nodes_.back();
    if (is_last != n.is_pipeline_output)
      err("there must be exactly one output node and it must be the last node in the graph.");
  }

  for (auto &n : nodes_) {
    if (n.is_batch_size_provider && !n.inputs.empty())
      err("a batch size provider cannot have inputs");
  }

  validated_ = true;
}

}  // namespace exec2
}  // namespace dali
