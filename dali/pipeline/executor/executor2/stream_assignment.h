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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_

#include <algorithm>
#include <optional>
#include <unordered_map>
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/exec2.h"


namespace dali {
namespace exec2 {

template <StreamPolicy policy>
class StreamAssignment;

inline bool NeedsStream(const ExecNode *node) {
  if (node->def) {
    if (node->def->op_type != OpType::CPU)
      return true;
  } else if (node->is_pipeline_output) {
    for (auto &pipe_out : node->inputs) {
      if (pipe_out->device == StorageDevice::GPU)
        return true;
    }
  }
  return false;
}

template <>
class StreamAssignment<StreamPolicy::Single> {
 public:
  StreamAssignment(ExecGraph &graph) {
    for (auto &node : graph.nodes) {
      if (NeedsStream(&node)) {
        needs_stream_ = true;
      }
    }
  }

  std::optional<int> GetStreamIdx(const ExecNode *node) {
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
  StreamAssignment(ExecGraph &graph) {
    for (auto &node : graph.nodes) {
      if (NeedsStream(&node)) {
        needs_stream_ = true;
      }
    }
  }

  std::optional<int> GetStreamIdx(const ExecNode *node) {
    OpType type = OpType::CPU;
    if (node->is_pipeline_output) {
      for (auto &pipe_out : node->inputs) {
        if (pipe_out->device == StorageDevice::GPU) {
          auto producer_type = pipe_out->producer->def->op_type;
          if (producer_type == OpType::GPU) {
            type = OpType::GPU;
            break;
          } else if (producer_type == OpType::MIXED) {
            type = OpType::MIXED;
          }
        }
      }
    } else {
      type = node->def->op_type;
    }

    switch (type) {
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
  StreamAssignment(ExecGraph &graph) {
    Assign(graph);
  }

  std::optional<int> GetStreamIdx(const ExecNode *node) const {
    auto it = stream_assignment_.find(node);
    assert(it != stream_assignment_.end());
    return it->second;
  }

  /** Gets the total number of streams required to run independent operators on separate streams. */
  int NumStreams() const {
    return total_streams_;
  }

 private:
  void Assign(ExecGraph &graph) {
    int next_stream_id;
    for (auto &node : graph.nodes) {
      next_stream_id += Assign(&node, next_stream_id);
    }
    for (auto &kv : stream_assignment_)
      if (kv.second == -1)
        kv.second = std::nullopt;

    total_streams_ = next_stream_id;
  }

  int Assign(ExecNode *node, int next_stream_id) {
    /* The assignment algorithm.

    The function assigns stream indices in a depth-first fashion. This allows for easy reuse
    of a stream for direct successors (they're serialized anyway, so we can just use one
    stream for them, skipping synchronization later).
    The function returns the number of streams needed for parallel execution of independent
    GPU/Mixed operators.

    1. Assign the current stream id to the node, if it needs a stream.
    2. Go over all outputs, depth first.
       a. recursively call Assign on a child node
          - if a child node already has assignment, it'll be skipped and return 0 streams needed.
       b. if processing an output needs some streams, bump the stream index.
       c. report the number of streams needed - if the node is a GPU/Mixed node, it'll need
          at least 1 stream (see the final return statement).

    Example (C denotes a CPU node, G - a GPU node)

    Graph:
                 ----C
                /
    --- C ---- G ---- G --- G
        \ \     \          /
         \ \     ----- G -
          \ \         /
           \ ------ C ------ G --- G
            \                 \
              ----- G --- C    ---- G

    --- G

    Visiting order (excludes edges leading to nodes already visited)
                 ----2
                /
    --- 0 ---- 1 ---- 3 --- 4
        \ \     \
         \ \     ----- 5
          \ \
           \ ------ 6 ------ 7 --- 8
            \                 \
              ---- 10 --- 11   ---- 9

    --- 12

    Return values marked on edges

                 --0-C
                /
    -5- C --2- G --1- G -1- G
        \ \     \
         \ \     ---1- G
          \ \
           \ ----2- C ---2-- G -1- G
            \                 \
              ---1- G -0- C    --1- G

    -1- G

    next_stream_id (includes CPU operators)

                 ---- 0
                /
    --- 0 ---- 0 ---- 0 --- 0
        \ \     \
         \ \     ----- 1
          \ \
           \ ------ 2 ------ 2 --- 2
            \                 \
              ----- 4 --- 4    ---- 3

    --- 5

    The final stream assignment is equal to next_stream_id shown above, but CPU operators get -1.
    */
    auto &assignment = stream_assignment_[node];
    if (assignment.has_value())  // this doubles as a visit marker
      return 0;
    bool needs_stream = NeedsStream(node);
    if (needs_stream) {
      assignment = next_stream_id;
    } else {
      assignment = -1;
    }

    int subgraph_streams = 0;
    for (auto *edge : node->outputs) {
      subgraph_streams += Assign(edge->consumer, next_stream_id + subgraph_streams);
    }

    return std::max(subgraph_streams, needs_stream ? 1 : 0);
  }

  std::unordered_map<const ExecNode *, std::optional<int>> stream_assignment_;
  int total_streams_ = 0;
};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
