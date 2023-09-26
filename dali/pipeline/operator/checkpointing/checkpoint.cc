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

#include "dali/pipeline/dali.pb.h"

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

Index Checkpoint::NumOp() const {
  return cpts_.size();
}

std::string Checkpoint::SerializeToProtobuf(const OpGraph &graph) const {
  dali_proto::Checkpoint checkpoint;
  const auto &nodes = graph.GetOpNodes();
  for (size_t i = 0; i < nodes.size(); i++) {
    auto op_cpt = checkpoint.add_cpts();
    op_cpt->set_operator_name(cpts_[i].OperatorName());
    op_cpt->set_operator_state(nodes[i].op->SerializeCheckpoint(cpts_[i]));
  }
  return checkpoint.SerializeAsString();
}

void Checkpoint::DeserializeFromProtobuf(const OpGraph &graph,
                                         const std::string &serialized_data) {
  Build(graph);
  dali_proto::Checkpoint checkpoint;
  checkpoint.ParseFromString(serialized_data);
  DALI_ENFORCE(checkpoint.cpts_size() == static_cast<int>(cpts_.size()),
               "The number of operators in the checkpoint differs from the number "
               "of operators in the pipeline. ");

  const auto &nodes = graph.GetOpNodes();
  for (int i = 0; i < checkpoint.cpts_size(); i++) {
    auto &op_cpt = cpts_[i];
    const auto &name = checkpoint.cpts(i).operator_name();
    const auto &data = checkpoint.cpts(i).operator_state();
    DALI_ENFORCE(name == op_cpt.OperatorName(),
                 "Attempted to restore state from checkpoint of another operator. "
                 "The checkpoint might come from another pipeline. ");
    nodes[i].op->DeserializeCheckpoint(op_cpt, data);
  }
}

}  // namespace dali
