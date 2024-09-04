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

  // TODO(michalz): implement minimal assignment for PerOperator policy
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

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_STREAM_ASSIGNMENT_H_
