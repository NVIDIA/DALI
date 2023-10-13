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

#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"

#include <string>
#include <vector>

#include "dali/pipeline/dali.pb.h"

#include "dali/operators/reader/loader/file_label_loader.h"

namespace dali {

namespace {

template<class T>
std::string SerializeToString(const T &obj) {
  // Only random number generators should use this,
  // as there is no other way to extract/restore their state.
  static_assert(std::is_same_v<T, std::mt19937> ||
                std::is_same_v<T, std::mt19937_64> ||
                std::is_same_v<T, std::default_random_engine>);
  std::stringstream stream;
  stream << obj;
  return stream.str();
}

template<class T>
T DeserializeFromString(const std::string &data) {
  // Only random number generators should use this,
  // as there is no other way to extract/restore their state.
  static_assert(std::is_same_v<T, std::mt19937> ||
                std::is_same_v<T, std::mt19937_64> ||
                std::is_same_v<T, std::default_random_engine>);
  std::stringstream stream(data);
  T obj;
  stream >> obj;
  return obj;
}

template<class T>
std::string SerializeRNG(const std::vector<T> &snapshot) {
  dali_proto::RNGSnapshotCPU proto_snapshot;
  for (const auto &rng : snapshot)
    proto_snapshot.add_rng(SerializeToString(rng));
  return proto_snapshot.SerializeAsString();
}

template<class T>
std::vector<T> DeserializeRNG(const std::string &data) {
  dali_proto::RNGSnapshotCPU proto_snapshot;
  proto_snapshot.ParseFromString(data);
  std::vector<T> snapshot;
  for (int i = 0; i < proto_snapshot.rng_size(); i++)
    snapshot.push_back(DeserializeFromString<T>(proto_snapshot.rng(i)));
  return snapshot;
}

}  // namespace

std::string SnapshotSerializer::Serialize(const std::vector<std::mt19937> &snapshot) {
  return SerializeRNG(snapshot);
}

template<> DLL_PUBLIC
std::vector<std::mt19937> SnapshotSerializer::Deserialize(const std::string &data) {
  return DeserializeRNG<std::mt19937>(data);
}

std::string SnapshotSerializer::Serialize(const std::vector<std::mt19937_64> &snapshot) {
  return SerializeRNG(snapshot);
}

template<> DLL_PUBLIC
std::vector<std::mt19937_64> SnapshotSerializer::Deserialize(const std::string &data) {
  return DeserializeRNG<std::mt19937_64>(data);
}

std::string SnapshotSerializer::Serialize(const LoaderBaseStateSnapshot &snapshot) {
  dali_proto::LoaderBaseStateSnapshot proto_snapshot;
  
  proto_snapshot.set_initial_buffer_filled(snapshot.initial_buffer_filled);
  for (const auto &sample : snapshot.samples_in_buffer) {
    proto_snapshot.add_samples_in_buffer(sample);
  }

  proto_snapshot.set_returned_sample_counter(snapshot.returned_sample_counter);
  proto_snapshot.set_read_sample_counter(snapshot.read_sample_counter);
  
  proto_snapshot.set_virtual_shard_id(snapshot.virtual_shard_id);
  for (const auto &shard : snapshot.shards) {
    auto proto_shard = proto_snapshot.add_shards();
    proto_shard->set_start(shard.start);
    proto_shard->set_end(shard.end);
  }
  
  proto_snapshot.set_virtual_shard_id(snapshot.virtual_shard_id);
  proto_snapshot.set_last_sample_idx(snapshot.last_sample_idx);

  proto_snapshot.set_rng(SerializeToString(snapshot.rng));
  proto_snapshot.set_seed(snapshot.seed);

  proto_snapshot.set_age(snapshot.age);
  
  return proto_snapshot.SerializeAsString();
}

std::string SnapshotSerializer::Serialize(const ExtraSnapshotData &snapshot) {
  dali_proto::ExtraSnapshotData proto_snapshot;
  proto_snapshot.set_current_epoch(snapshot.current_epoch);
  return proto_snapshot.SerializeAsString();
}

std::string SnapshotSerializer::Serialize(const LoaderStateSnapshot &snapshot) {
  dali_proto::LoaderStateSnapshot proto_snapshot;
  proto_snapshot.set_base(Serialize(snapshot.base));
  proto_snapshot.set_extra(Serialize(snapshot.extra));
  return proto_snapshot.SerializeAsString();
}

template<> DLL_PUBLIC
LoaderBaseStateSnapshot SnapshotSerializer::Deserialize(const std::string &data) {
  dali_proto::LoaderBaseStateSnapshot proto_snapshot;
  proto_snapshot.ParseFromString(data);
  LoaderBaseStateSnapshot snapshot;

  snapshot.initial_buffer_filled = proto_snapshot.initial_buffer_filled();
  snapshot.samples_in_buffer.reserve(proto_snapshot.samples_in_buffer_size());
  for (const auto &sample : proto_snapshot.samples_in_buffer()) {
    snapshot.samples_in_buffer.push_back(sample);
  }

  snapshot.returned_sample_counter = proto_snapshot.returned_sample_counter();
  snapshot.read_sample_counter = proto_snapshot.read_sample_counter();

  snapshot.virtual_shard_id = proto_snapshot.virtual_shard_id();
  snapshot.shards.reserve(proto_snapshot.shards_size());
  for (const auto &shard : proto_snapshot.shards()) {
    snapshot.shards.push_back({shard.start(), shard.end()});
  }

  snapshot.has_last_sample = proto_snapshot.has_last_sample();
  snapshot.last_sample_idx = proto_snapshot.last_sample_idx();

  snapshot.rng = DeserializeFromString<decltype(snapshot.rng)>(proto_snapshot.rng());
  snapshot.seed = proto_snapshot.seed();

  snapshot.age = proto_snapshot.age();

  return snapshot;
}

template<> DLL_PUBLIC
ExtraSnapshotData SnapshotSerializer::Deserialize(const std::string &data) {
  dali_proto::ExtraSnapshotData proto_snapshot;
  proto_snapshot.ParseFromString(data);
  return {proto_snapshot.current_epoch()};
}

template<> DLL_PUBLIC
LoaderStateSnapshot SnapshotSerializer::Deserialize(const std::string &data) {
  dali_proto::LoaderStateSnapshot proto_snapshot;
  LoaderStateSnapshot snapshot;
  proto_snapshot.ParseFromString(data);
  snapshot.base = Deserialize<LoaderBaseStateSnapshot>(proto_snapshot.base());
  snapshot.extra = Deserialize<ExtraSnapshotData>(proto_snapshot.extra());
  return snapshot;
}

}  // namespace dali
