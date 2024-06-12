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

#include <utility>

#include "dali/pipeline/operator/checkpointing/checkpoint.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

void Checkpoint::Clear() {
  cpts_.clear();
  name2id_.clear();
  iteration_id_ = 0;
}

int Checkpoint::AddOperator(std::string instance_name) {
  if (!name2id_.emplace(instance_name, cpts_.size()).second)
    DALI_FAIL("The checkpoint already contains an operator with name \"", instance_name, "\".");
  cpts_.emplace_back(std::move(instance_name));
  return cpts_.size() - 1;
}

std::optional<int> Checkpoint::OperatorIdx(std::string_view instance_name) const {
  auto it = name2id_.find(instance_name);
  if (it == name2id_.end())
    return std::nullopt;
  return it->second;
}

const OpCheckpoint &Checkpoint::GetOpCheckpoint(std::string_view instance_name) const {
  auto idx = OperatorIdx(instance_name);
  if (!idx)
    DALI_FAIL("There's no checkpoint for operator \"", instance_name, "\".");
  return GetOpCheckpoint(*idx);
}

OpCheckpoint &Checkpoint::GetOpCheckpoint(int idx) {
  DALI_ENFORCE_VALID_INDEX(idx, cpts_.size());
  return cpts_[idx];
}

const OpCheckpoint &Checkpoint::GetOpCheckpoint(int idx) const {
  DALI_ENFORCE_VALID_INDEX(idx, cpts_.size());
  return cpts_[idx];
}

void Checkpoint::SetIterationId(size_t iteration_id) {
  iteration_id_ = iteration_id;
}

size_t Checkpoint::GetIterationId() const {
  return iteration_id_;
}

void Checkpoint::SetOrder(AccessOrder order) {
  for (auto &cpt : cpts_)
    cpt.SetOrder(order);
}

Index Checkpoint::NumOp() const {
  return cpts_.size();
}

std::string Checkpoint::SerializeToProtobuf(ExecutorBase &exec) const {
  dali_proto::Checkpoint checkpoint;
  for (const OpCheckpoint &cpt : cpts_) {
    auto op_cpt = checkpoint.add_cpts();
    const auto &name = cpt.OperatorName();
    op_cpt->set_operator_name(name);
    auto *op = exec.GetOperator(name);
    assert(op);
    op_cpt->set_operator_state(op->SerializeCheckpoint(cpt));
  }
  checkpoint.mutable_external_ctx_cpt()->set_pipeline_data(external_ctx_cpt_.pipeline_data);
  checkpoint.mutable_external_ctx_cpt()->set_iterator_data(external_ctx_cpt_.iterator_data);
  return checkpoint.SerializeAsString();
}

void Checkpoint::DeserializeFromProtobuf(ExecutorBase &exec, const std::string &serialized_data) {
  Clear();
  dali_proto::Checkpoint checkpoint;
  checkpoint.ParseFromString(serialized_data);

  for (int i = 0; i < checkpoint.cpts_size(); i++) {
    const auto &name = checkpoint.cpts(i).operator_name();
    const auto &data = checkpoint.cpts(i).operator_state();
    auto *op = exec.GetOperator(name);
    DALI_ENFORCE(op, make_string(
                 "The executor doesn't recognize \"", name, "\" as a name of an operator.\n"
                 "The checkpoint might come from another pipeline."));
    auto idx = AddOperator(name);  // this extends the `cpts_` vector
    auto &op_cpt = cpts_[idx];
    op->DeserializeCheckpoint(op_cpt, data);
  }
  external_ctx_cpt_.pipeline_data = checkpoint.external_ctx_cpt().pipeline_data();
  external_ctx_cpt_.iterator_data = checkpoint.external_ctx_cpt().iterator_data();
}

}  // namespace dali
