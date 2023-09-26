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

std::string SnapshotSerializer::Serialize(const LoaderStateSnapshot &snapshot) {
  dali_proto::ReaderStateSnapshot proto_snapshot;
  proto_snapshot.mutable_loader_state()->set_rng(SerializeToString(snapshot.rng));
  proto_snapshot.mutable_loader_state()->set_current_epoch(snapshot.current_epoch);
  return proto_snapshot.SerializeAsString();
}

template<> DLL_PUBLIC
LoaderStateSnapshot SnapshotSerializer::Deserialize(const std::string &data) {
  dali_proto::ReaderStateSnapshot proto_snapshot;
  proto_snapshot.ParseFromString(data);
  return LoaderStateSnapshot {
    DeserializeFromString<std::default_random_engine>(proto_snapshot.loader_state().rng()),
    proto_snapshot.loader_state().current_epoch(),
  };
}

}  // namespace dali
