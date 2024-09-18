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
#include <map>
#include <optional>
#include <queue>
#include <unordered_map>
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
  if (node->is_pipeline_output || node->backend == OpType::CPU) {
    for (auto &input : node->inputs) {
      if (input->device == StorageDevice::GPU && !input->metadata)
        return true;
    }
    return false;
  } else {
    return true;
  }
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
      case OpType::CPU:
        if (NeedsStream(&node)) {  // treat CPU nodes with GPU inputs as GPU
          has_gpu_ = true;
          if (has_mixed_)
            return;
        }
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
   * CPU nodes get GPU stream if they need one (i.e. they have a GPU input)
   */
  std::optional<int> operator[](const ExecNode *node) const {
    switch (NodeType(node)) {
    case OpType::CPU:
      if (!NeedsStream(node))
        return std::nullopt;
      // fall-through to GPU
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
    std::map<int, int> ready_streams;
  };

  void Assign(ExecGraph &graph) {
    int num_nodes = graph.Nodes().size();

    // the nodes in the are already sorted topologically
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
    // Process the nodes in topological order.
    for (auto [node, meta] : sorted_nodes_) {
      ProcessNode(node, meta);
    }
  }

  void ProcessNode(const ExecNode *node, NodeMeta *meta) {
    /* The algorithm

    Each node has an associated NodeMeta, which contains the stream assignment and a set
    of "ready streams". This is the set of streams which are ready after this node is complete.
    It includes the streams of the producers + the stream of this node.
    Each stream is associated with a "use count" and a ready set contains, alongside the id,
    the value of the use count at the time at which the stream was inserted to the set.
    Later, when trying to reuse the ready streams, the streams which were inserted with an
    outdated use count are rejected.

    For each node (topologically sorted):

    1a. Pick the producers' ready stream with ths smallest id (reject streams with stale use count).
    1b. If there are no ready streams, bump the total number of streams and get a new stream id.

    2. Bump the use count of the currently assigned stream.

    3. Compute the current ready set as the union of input ready sets + the current stream id.

    When computing the union, stale streams are removed to speed up subsequent lookups.
    */

    std::optional<int> stream_id = {};

    for (auto &e : node->inputs) {
      auto &prod_meta = node_meta_[e->producer];
      // If we're a GPU contributor (and therefore we need a stream assignment), check if the
      // producer's ready stream set contains something.
      if (meta->gpu_contributor) {
        for (auto it = prod_meta.ready_streams.begin(); it != prod_meta.ready_streams.end(); ++it) {
          auto [id, use_count] = *it;
          if (use_count < stream_use_count_[id]) {
            continue;
          }
          if (!stream_id.has_value() || *stream_id > id)
            stream_id = id;
        }
      }
      // Add producer's ready set to the current one.
      CombineReady(*meta, prod_meta);
    }

    if (stream_id.has_value()) {
      UseStream(*meta, *stream_id);
    } else {
      if (meta->needs_stream) {
        stream_id = total_streams_++;
        stream_use_count_[*stream_id] = 1;
        UseStream(*meta, *stream_id);
      }
    }

    assert(!stream_id.has_value() || meta->ready_streams.count(*stream_id));
  }

  void UseStream(NodeMeta &meta, int id) {
    int use_count = ++stream_use_count_[id];
    meta.stream_id = id;
    meta.ready_streams[id] = use_count;
  }

  void CombineReady(NodeMeta &to, NodeMeta &from) {
    for (auto it = from.ready_streams.begin(); it != from.ready_streams.end(); ) {
      auto [id, old_use_count] = *it;
      int current_use_count = stream_use_count_[id];
      if (current_use_count > old_use_count) {
        it = from.ready_streams.erase(it);
        continue;
      }
      to.ready_streams[id] = current_use_count;
      ++it;
    }
  }

  int total_streams_ = 0;

  std::unordered_map<const ExecNode *, NodeMeta> node_meta_;
  std::unordered_map<int, int> stream_use_count_;
  std::vector<std::pair<const ExecNode *, NodeMeta *>> sorted_nodes_;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
