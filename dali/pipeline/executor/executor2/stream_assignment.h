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
#include <cassert>
#include <functional>
#include <optional>
#include <queue>
#include <unordered_map>
#include <set>
#include <utility>
#include <vector>
#include "dali/pipeline/graph/graph_util.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
// TODO(michalz): This is here for review process only. Remove when exec2.h is available
// #include "dali/pipeline/executor/executor2/exec2.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {

// TODO(michalz): This is here for review process only. Remove when exec2.h is available
enum class StreamPolicy : int {
  Single,       //< There's just one stream that's used by all operators
  PerBackend,   //< Operators are scheduled on a stream specific to their backend (mixed or GPU)
  PerOperator   //< Independent operators are executed on separate streams.
};


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

/** A trivial stream policy, with just one stream shared by all non-CPU operaotrs. */
template <>
class StreamAssignment<StreamPolicy::Single> {
 public:
  explicit StreamAssignment(ExecGraph &graph) {
    for (auto &node : graph.Nodes()) {
      if (NeedsStream(&node)) {
        needs_stream_ = true;
        break;
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


/** A simple stream policy where all mixed and GPU operators share their respective streams.
 *
 * In this policy there are 0..2 streams, depending on the number of mixed and GPU nodes:
 * 0 - only CPU nodes
 * 1 - there are some mixed or some GPU nodes, but not both
 * 2 - there are both mixed and CPU nodes present.
 */
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
    auto it = node_meta_.find(node);
    assert(it != node_meta_.end());
    return it->second.stream_id;
  }

  /** Gets the total number of streams required to run independent operators on separate streams. */
  int NumStreams() const {
    return total_streams_;
  }

 private:
  struct NodeMeta {
    int index;  // index in the sorted_nodes_
    bool needs_stream = false;     // whether this exact node needs a stream
    bool gpu_contributor = false;  // whether the node contributes to a GPU operator
    std::optional<int> stream_id;
    std::set<int> ready_streams;
  };

  void Assign(ExecGraph &graph) {
    int num_nodes = graph.Nodes().size();

    // the nodes in the graph must be sorted topologically
    sorted_nodes_.reserve(num_nodes);
    for (auto &node : graph.Nodes()) {
      int idx = sorted_nodes_.size();
      NodeMeta &meta = node_meta_.insert({ &node, {} }).first->second;
      meta.index = idx;
      sorted_nodes_.push_back({ &node, &meta });
    }

    assert(static_cast<size_t>(num_nodes) == sorted_nodes_.size());

    FindGPUContributors(graph);
    RunAssignment(graph);
    ClearCPUStreams();
  }


  void ClearCPUStreams() {
    for (auto &[node, meta] : node_meta_)
      if (!meta.needs_stream)
        meta.stream_id = std::nullopt;
  }


  void FindGPUContributors(ExecGraph &graph) {
    // Run DFS, output to input, and find nodes which contribute to any node that requires a stream
    graph::ClearVisitMarkers(graph.Nodes());
    for (auto it = graph.Nodes().rbegin(); it != graph.Nodes().rend(); ++it) {
      auto &node = *it;
      FindGPUContributors(&node, false);
    }
  }

  void FindGPUContributors(const ExecNode *node, bool is_gpu_contributor) {
    graph::Visit v(node);
    if (!v)
      return;
    auto &meta = node_meta_[node];
    meta.needs_stream = NeedsStream(node);
    if (!is_gpu_contributor)
      is_gpu_contributor = meta.needs_stream;
    if (is_gpu_contributor)
      meta.gpu_contributor = true;
    for (auto *inp : node->inputs)
      FindGPUContributors(inp->producer, is_gpu_contributor);
  }

  void RunAssignment(ExecGraph &graph) {
    graph::ClearVisitMarkers(graph.Nodes());
    for (int i = sorted_nodes_.size() - 1; i >= 0; i--) {
      ProcessNode(sorted_nodes_[i].first, sorted_nodes_[i].second);
    }
  }

  void ProcessNode(const ExecNode *node, NodeMeta *meta) {
    graph::Visit v(node);
    if (!v)
      return;

    auto &stream_id = meta->stream_id;
    assert(!stream_id.has_value());

    for (auto &e : node->inputs) {
      auto &prod_meta = node_meta_[e->producer];
      ProcessNode(e->producer, &prod_meta);
      if (meta->gpu_contributor && !prod_meta.ready_streams.empty()) {
        if (!stream_id.has_value() || *stream_id > *prod_meta.ready_streams.begin()) {
          stream_id = *prod_meta.ready_streams.begin();
        }
      }
      meta->ready_streams.insert(prod_meta.ready_streams.begin(), prod_meta.ready_streams.end());
    }

    if (stream_id.has_value()) {
      for (auto &e : node->inputs) {
        EraseReady(e->producer, *stream_id, node);
      }
    } else {
      if (meta->needs_stream) {
        stream_id = total_streams_++;
        meta->ready_streams.insert(*stream_id);
      }
    }

    assert(!stream_id.has_value() || meta->ready_streams.count(*stream_id));
  }

  void EraseReady(const ExecNode *node, int id, const ExecNode *sentinel) {
    if (node == sentinel)
      return;
    auto &meta = node_meta_[node];
    if (meta.ready_streams.erase(id)) {
      for (auto &e : node->inputs)
        EraseReady(e->producer, id, sentinel);
      for (auto &out : node->outputs)
        for (auto &e : out.consumers)
          EraseReady(e->consumer, id, sentinel);
    }
  }

  int total_streams_ = 0;

  std::unordered_map<const ExecNode *, NodeMeta> node_meta_;
  std::vector<std::pair<const ExecNode *, NodeMeta *>> sorted_nodes_;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
