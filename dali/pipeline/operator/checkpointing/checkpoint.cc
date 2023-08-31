// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/checkpointing/checkpoint.h"
#include "dali/pipeline/graph/op_graph.h"

namespace dali {

void Checkpoint::Build(const OpGraph &graph) {
  cpts_.reserve(graph.GetOpNodes().size());
  for (const auto &node : graph.GetOpNodes())
    cpts_.emplace_back(node.spec);
}

OpCheckpoint &Checkpoint::GetOpCheckpoint(OpNodeId id) {
  DALI_ENFORCE_VALID_INDEX(id, cpts_.size());
  return cpts_[id];
}

const OpCheckpoint &Checkpoint::GetOpCheckpoint(OpNodeId id) const {
  DALI_ENFORCE_VALID_INDEX(id, cpts_.size());
  return cpts_[id];
}

void Checkpoint::SetIterationId(size_t iteration_id) {
  iteration_id_ = iteration_id;
}

size_t Checkpoint::GetIterationId() const {
  return iteration_id_;
}

}  // namespace dali
