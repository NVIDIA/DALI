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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_

#include <algorithm>
#include <functional>
#include <optional>
#include <queue>
#include <unordered_map>
#include <set>
#include <utility>
#include <vector>
#include "dali/pipeline/graph/graph_util.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/exec2.h"

namespace dali {
namespace exec2 {

template <StreamPolicy policy>
class StreamAssignment;

inline bool NeedsStream(const ExecNode *node) {
  if (node->is_pipeline_output) {
    for (auto &pipe_out : node->inputs) {
      if (pipe_out->device == StorageDevice::GPU)
        return true;
    }
  } else {
    return node->backend != OpType::CPU;
  }
  return false;
}

inline OpType NodeType(const ExecNode *node) {
  if (node->is_pipeline_output) {
    OpType type = OpType::CPU;
    for (auto &pipe_out : node->inputs) {
      if (pipe_out->device == StorageDevice::GPU) {
        auto producer_type = pipe_out->producer->backend;
        if (producer_type == OpType::GPU) {
          return OpType::GPU;
        } else if (producer_type == OpType::MIXED) {
          type = OpType::MIXED;
        }
      }
    }
    return type;
  } else {
    return node->backend;
  }
}

template <>
class StreamAssignment<StreamPolicy::Single> {
 public:
  explicit StreamAssignment(ExecGraph &graph) {
    for (auto &node : graph.Nodes()) {
      if (NeedsStream(&node)) {
        needs_stream_ = true;
      }
    }
  }

  std::optional<int> operator[](const ExecNode *node) const {
    if (NeedsStream(node))
      return 0;
    else
      return std::nullopt;
  }

  int NumStreams() const {
    return needs_stream_ ? 1 : 0;
  }

 private:
  bool needs_stream_ = false;
};


template <>
class StreamAssignment<StreamPolicy::PerBackend> {
 public:
  explicit StreamAssignment(ExecGraph &graph) {
    for (auto &node : graph.Nodes()) {
      switch (NodeType(&node)) {
      case OpType::GPU:
        has_gpu_ = true;
        if (has_mixed_)
          return;  // we already have both, nothing more can happen
        break;
      case OpType::MIXED:
        has_mixed_ = true;
        if (has_gpu_)
          return;  // we already have both, nothing more can happen
        break;
      default:
        break;
      }
    }
  }

  /** Returns a stream index for a non-CPU operator.
   *
   * If the node is a Mixed node, it gets stream index 0.
   * If the node is a GPU node it gets stream index 1 if there are any mixed nodes, otherwise
   * the only stream is the GPU stream and the returned index is 0.
   */
  std::optional<int> operator[](const ExecNode *node) const {
    switch (NodeType(node)) {
    case OpType::CPU:
      return std::nullopt;
    case OpType::GPU:
      return has_mixed_ ? 1 : 0;
    case OpType::MIXED:
      return 0;
    default:
      assert(false && "Unreachable");
      return std::nullopt;
    }
  }

  int NumStreams() const {
    return has_gpu_ + has_mixed_;
  }

 private:
  bool has_gpu_ = false;
  bool has_mixed_ = false;
};

/** Implements per-operator stream assignment.
 *
 * This policy implements stream assingment such that independent GPU/Mixed operators get
 * separate streams. When there's a dependency then one dependent operator shares the stream of
 * its predecessor.
 *
 * Example - numbers are stream indices, "X" means no stream, "s" means synchronization
 * ```
 * CPU(X) ---- GPU(0) --- GPU(0) -- GPU(0) -- output 0
 *              \                    s
 *               \                  /
 *                ----- GPU(1) ----
 *                        \
 *                         \
 * CPU(X) --- GPU(2) ----s GPU(1) ----------s output 1
 * ```
 */
template <>
class StreamAssignment<StreamPolicy::PerOperator> {
 public:
  explicit StreamAssignment(ExecGraph &graph) {
    Assign(graph);
  }

  std::optional<int> operator[](const ExecNode *node) const {
    auto it = node_ids_.find(node);
    assert(it != node_ids_.end());
    return stream_assignment_[it->second];
  }

  /** Gets the total number of streams required to run independent operators on separate streams. */
  int NumStreams() const {
    return total_streams_;
  }

 private:
  void Assign(ExecGraph &graph) {
    // pre-fill the id pool with sequential numbers
    for (int i = 0, n = graph.Nodes().size(); i < n; i++) {
      free_stream_ids_.insert(i);
    }

    // Sort the graph topologically with DFS
    for (auto &node : graph.Nodes()) {
      Sort(&node);
    }

    for (auto &node : graph.Nodes()) {
      if (node.inputs.empty())
        queue_.push({ node_ids_[&node], NextStreamId(&node).value_or(kInvalidStreamIdx) });
    }

    assert(graph.Nodes().size() == sorted_nodes_.size());
    stream_assignment_.resize(sorted_nodes_.size());

    FindGPUContributors(graph);

    graph::ClearVisitMarkers(graph.Nodes());
    Traverse();
    ClearCPUStreams();
    total_streams_ = CalcNumStreams();
  }

  void Traverse() {
    while (!queue_.empty()) {
      // PrintQueue(); /* uncomment for debugging */
      auto [idx, stream_idx] = queue_.top();
      std::optional<int> stream_id;
      if (stream_idx != kInvalidStreamIdx)
        stream_id = stream_idx;

      queue_.pop();
      auto *node = sorted_nodes_[idx];
      // This will be true for nodes which has no outputs or which doesn't contribute to any
      // GPU nodes.
      bool keep_stream_id = stream_id.has_value();

      graph::Visit v(node);
      if (!v) {
        assert(stream_assignment_[idx].value_or(kInvalidStreamIdx) <= stream_idx);
        continue;  // we've been here already - skip
      }

      stream_assignment_[idx] = stream_id;

      if (stream_id.has_value())
        free_stream_ids_.insert(*stream_id);
      for (auto &output_desc : node->outputs) {
        for (auto *out : output_desc.consumers) {
          auto out_stream_id = NextStreamId(out->consumer, stream_id);
          if (out_stream_id.has_value())
            keep_stream_id = false;
          queue_.push({node_ids_[out->consumer], out_stream_id.value_or(kInvalidStreamIdx)});
        }
      }
      if (keep_stream_id)
        free_stream_ids_.erase(*stream_id);
    }
  }

  void ClearCPUStreams() {
    for (int i = 0, n = sorted_nodes_.size(); i < n; i++) {
      if (!NeedsStream(sorted_nodes_[i]))
        stream_assignment_[i] = std::nullopt;
    }
  }

  int CalcNumStreams() {
    int max = -1;
    for (auto a : stream_assignment_) {
      if (a.has_value())
        max = std::max(max, *a);
    }
    return max + 1;
  }

  void PrintQueue(std::ostream &os = std::cout) {
    auto q2 = queue_;
    while (!q2.empty()) {
      auto [idx, stream_idx] = q2.top();
      q2.pop();
      auto *node = sorted_nodes_[idx];
      if (!node->instance_name.empty())
        os << node->instance_name;
      else if (node->is_pipeline_output)
        os << "<output>";
      else
        os << "[" << idx << "]";
      os << "(";
      if (stream_idx != kInvalidStreamIdx)
        os << stream_idx;
      else
        os << "none";
      os << ") ";
    }
    os << "\n";
  }

  void Sort(const ExecNode *node) {
    assert(node);
    graph::Visit visit(node);
    if (!visit)
      return;
    int idx = sorted_nodes_.size();
    node_ids_.emplace(node, idx);
    for (auto *edge : node->inputs) {
      assert(edge);
      Sort(edge->producer);
    }
    sorted_nodes_.push_back(node);
  }

  std::optional<int> NextStreamId(const ExecNode *node,
                                  std::optional<int> prev_stream_id = std::nullopt) {
    // If the preceding node had a stream, then we have to pass it on through CPU nodes
    // if there are any GPU nodes down the graph.
    // If the preciding node didn't have a stream, then we only need a stream if current
    // node needs a stram.
    bool needs_stream = prev_stream_id.has_value()
                      ? gpu_contributors_.count(node) != 0
                      : NeedsStream(node);
    if (needs_stream) {
      assert(!free_stream_ids_.empty());
      auto b = free_stream_ids_.begin();
      int ret = *b;
      free_stream_ids_.erase(b);
      return ret;
    } else {
      return std::nullopt;
    }
  }

  void FindGPUContributors(ExecGraph &graph) {
    // Run DFS, output to input, and find nodes which contribute to any node that requires a stream
    graph::ClearVisitMarkers(graph.Nodes());
    for (auto &node : graph.Nodes()) {
      if (node.outputs.empty())
        FindGPUContributors(&node, false);
    }
  }

  void FindGPUContributors(const ExecNode *node, bool is_gpu_contributor) {
    graph::Visit v(node);
    if (!v)
      return;
    if (!is_gpu_contributor)
      is_gpu_contributor = NeedsStream(node);
    if (is_gpu_contributor)
      gpu_contributors_.insert(node);
    for (auto *inp : node->inputs)
      FindGPUContributors(inp->producer, is_gpu_contributor);
  }


  static constexpr int kInvalidStreamIdx = 0x7fffffff;
  std::vector<std::optional<int>> stream_assignment_;
  int total_streams_ = 0;
  std::unordered_map<const ExecNode *, int> node_ids_;  // topologically sorted nodes
  std::set<const ExecNode *> gpu_contributors_;
  std::vector<const ExecNode *> sorted_nodes_;
  std::set<int> free_stream_ids_;
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> queue_;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
