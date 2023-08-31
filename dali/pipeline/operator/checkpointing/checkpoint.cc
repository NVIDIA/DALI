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

#include <string>

#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

namespace {

template<class T> std::string SerializeToString(const T &obj) {
  std::stringstream stream;
  stream << obj;
  return stream.str();
}

template<class T> T DeserializeFromString(const std::string &str) {
  std::stringstream stream(str);
  T obj;
  stream >> obj;
  return obj;
}

/**
 * @brief Helper object to be used with std::visit for CheckpointingData serialization
*/
struct CheckpointDataToProto {
  explicit CheckpointDataToProto(dali_proto::Checkpoint_OpCheckpoint *proto) : proto_(proto) {}

  void operator()(std::monostate) {
    /* set the oneof field to stateless */
    proto_->mutable_stateless();
  }

  void operator()(const RNGSnapshotCPU &data) {
    auto snapshot = proto_->mutable_rng_cpu();
    for (const auto &rng : data.rng)
      snapshot->add_rng(SerializeToString(data.rng));
  }

  void operator()(const RNGSnapshotCPU64 &data) {
    auto snapshot = proto_->mutable_rng_cpu64();
    for (const auto &rng : data.rng)
      snapshot->add_rng(SerializeToString(data.rng));
  }

  void operator()(const ReaderStateSnapshot &data) {
    auto snapshot = proto_->mutable_reader();
    snapshot->mutable_loader_state()->set_rng(SerializeToString(data.loader_state.rng));
    snapshot->mutable_loader_state()->set_current_epoch(data.loader_state.current_epoch);
  }

  void operator()(const DummySnapshot &data) {
    auto snapshot = proto_->mutable_dummy();
    for (auto x : data.dummy_state)
      snapshot->add_dummy_state(x);
  }

 private:
  dali_proto::Checkpoint_OpCheckpoint *proto_;
};

void ToProto(const OpCheckpoint &cpt, dali_proto::Checkpoint_OpCheckpoint *proto) {
  proto->set_operator_name(cpt.OperatorName());
  std::visit(CheckpointDataToProto(proto), cpt.GetCheckpointingData());
}

RNGSnapshotCPU ToRngCpu(const dali_proto::Checkpoint_OpCheckpoint &proto) {
  RNGSnapshotCPU snapshot;
  auto &data = proto.rng_cpu();
  for (int i = 0; i < data.rng_size(); i++)
    snapshot.rng.push_back(DeserializeFromString<std::mt19937>(data.rng(i)));
  return snapshot;
}

RNGSnapshotCPU64 ToRngCpu64(const dali_proto::Checkpoint_OpCheckpoint &proto) {
  RNGSnapshotCPU64 snapshot;
  auto &data = proto.rng_cpu();
  for (int i = 0; i < data.rng_size(); i++)
    snapshot.rng.push_back(DeserializeFromString<std::mt19937_64>(data.rng(i)));
  return snapshot;
}

ReaderStateSnapshot ToReader(const dali_proto::Checkpoint_OpCheckpoint &proto) {
  auto &data = proto.reader();
  DALI_ENFORCE(data.has_loader_state(),
               "Serialized operator state is missing the `loader_state` field. ");
  DALI_ENFORCE(data.loader_state().has_rng(),
              "Serialized operator state is missing the `rng` field. ");
  DALI_ENFORCE(data.loader_state().has_current_epoch(),
              "Serialized operator state is missing the `current_epoch` field. ");
  return ReaderStateSnapshot {
    LoaderStateSnapshot {
      DeserializeFromString<std::default_random_engine>(data.loader_state().rng()),
      data.loader_state().current_epoch(),
    }
  };
}

DummySnapshot ToDummy(const dali_proto::Checkpoint_OpCheckpoint &proto) {
  DummySnapshot snapshot;
  auto &data = proto.dummy();
  for (int i = 0; i < data.dummy_state_size(); i++)
    snapshot.dummy_state.push_back(data.dummy_state(i));
  return snapshot;
}

OpCheckpoint FromProto(const dali_proto::Checkpoint_OpCheckpoint &proto) {
  OpCheckpoint cpt(proto.operator_name());
  switch (proto.data_case()) {
    case dali_proto::Checkpoint_OpCheckpoint::kStateless:
      /* leave the default empty checkpoint */
      break;

    case dali_proto::Checkpoint_OpCheckpoint::kRngCpu:
      cpt.MutableCheckpointState() = ToRngCpu(proto);
      break;

    case dali_proto::Checkpoint_OpCheckpoint::kRngCpu64:
      cpt.MutableCheckpointState() = ToRngCpu64(proto);
      break;

    case dali_proto::Checkpoint_OpCheckpoint::kReader:
      cpt.MutableCheckpointState() = ToReader(proto);
      break;

    case dali_proto::Checkpoint_OpCheckpoint::kDummy:
      cpt.MutableCheckpointState() = ToDummy(proto);
      break;

    default:
      DALI_FAIL("Parsing checkpoint data failed. Unknown data type found. ");
  }

  return cpt;
}

}  // namespace

void Checkpoint::Build(const OpGraph &graph) {
  DALI_ENFORCE(cpts_.empty(), "Checkpoint should be built only once. ");
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

std::string Checkpoint::SerializeToProtobuf() const {
  dali_proto::Checkpoint proto;
  for (const auto &cpt : cpts_)
    ToProto(cpt, proto.add_cpts());
  return proto.SerializeAsString();
}

void Checkpoint::DeserializeFromProtobuf(const std::string &serialized_data) {
  dali_proto::Checkpoint proto;
  proto.ParseFromString(serialized_data);
  for (int i = 0; i < proto.cpts_size(); i++)
    cpts_.push_back(FromProto(proto.cpts(i)));
}

}  // namespace dali
