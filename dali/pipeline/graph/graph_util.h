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

#ifndef DALI_PIPELINE_GRAPH_GRAPH_UTIL_H_
#define DALI_PIPELINE_GRAPH_GRAPH_UTIL_H_

#include <stdexcept>
#include "dali/core/util.h"

namespace dali {
namespace graph {

IMPL_HAS_MEMBER(visit_pending);

/** A helper for visiting DAG nodes.
 *
 * When the node has a "visit_pending" member, the helper detects cycles.
 */
template <typename Node>
class Visit {
 public:
  explicit Visit(Node *n) : node_(n) {
    if constexpr (has_member_visit_pending_v<Node>) {
      if (node_->visit_pending)
        throw std::logic_error("Cycle detected.");
      node_->visit_pending = true;
    }
    new_visit_ = !n->visited;
    node_->visited = true;
  }
  ~Visit() {
    if constexpr (has_member_visit_pending_v<Node>) {
      node_->visit_pending = false;
    }
  }

  explicit operator bool() const {
    return new_visit_;
  }

 private:
  Node *node_;
  bool new_visit_;
};

/** Clears visit markers.
 *
 * Sets the `visited` field to `false`.
 * Typically used at the beginning of a graph-processing algorithm.
 */
template <typename NodeList>
static void ClearVisitMarkers(NodeList &nodes) {
  for (auto &node : nodes)
    node.visited = false;
}

}  // namespace graph
}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_GRAPH_UTIL_H_
