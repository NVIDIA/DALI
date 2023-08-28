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
#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

namespace detail {
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

  struct CheckpointDataToProto {
    CheckpointDataToProto(dali_proto::Checkpoint_OpCheckpoint *proto) : proto_(proto) {}

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

    void operator()(const LoaderStateSnapshot &data) {
      auto snapshot = proto_->mutable_loader();
      snapshot->set_rng(SerializeToString(data.rng));
      snapshot->set_current_epoch(data.current_epoch);
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

  LoaderStateSnapshot ToLoader(const dali_proto::Checkpoint_OpCheckpoint &proto) {
    auto &data = proto.loader();
    return LoaderStateSnapshot {
      DeserializeFromString<std::default_random_engine>(data.rng()),
      data.current_epoch(),
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
    switch(proto.data_case()) {
      case dali_proto::Checkpoint_OpCheckpoint::kRngCpu:
        cpt.MutableCheckpointState() = ToRngCpu(proto);
        break;

      case dali_proto::Checkpoint_OpCheckpoint::kRngCpu64:
        cpt.MutableCheckpointState() = ToRngCpu64(proto);
        break;

      case dali_proto::Checkpoint_OpCheckpoint::kLoader:
        cpt.MutableCheckpointState() = ToLoader(proto);
        break;

      case dali_proto::Checkpoint_OpCheckpoint::kDummy:
        cpt.MutableCheckpointState() = ToDummy(proto);
        break;

      default:
        DALI_FAIL("Parsing checkpoint data failed. Unknown data type found. ");
    }

    return cpt;
  }
}

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

void Checkpoint::SerializeToFile(const std::string &path) const {
  dali_proto::Checkpoint proto;
  for (const auto &cpt : cpts_)
    detail::ToProto(cpt, proto.add_cpts());

  std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!proto.SerializeToOstream(&output))
    DALI_FAIL("Saving the checkpoint to file failed. ");
}

void Checkpoint::DeserializeFromFile(const std::string &path) {
  dali_proto::Checkpoint proto;

  std::fstream input(path, std::ios::in | std::ios::binary);
  if (!input) {
    DALI_FAIL(make_string(path, ": File not found. "));
  } else if (!proto.ParseFromIstream(&input)) {
    DALI_FAIL("Failed to parse checkpoint from file. ");
  }

  for (int i = 0; i < proto.cpts_size(); i++)
    cpts_.push_back(detail::FromProto(proto.cpts(i)));
}

}  // namespace dali
