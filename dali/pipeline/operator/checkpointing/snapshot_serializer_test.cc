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

#include <gtest/gtest.h>

#include "dali/test/dali_test.h"

namespace dali {

class SnapshotSerializerTest : public DALITest {};

TEST_F(SnapshotSerializerTest, VectorMt19937) {
  std::vector<std::mt19937> snapshot;
  for (int i = 123; i <= 321; i++)
    snapshot.emplace_back(i);

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<std::mt19937>>(serialized);

  ASSERT_EQ(snapshot.size(), deserialized.size());
  for (size_t i = 0; i < snapshot.size(); i++)
    EXPECT_EQ(snapshot[i], deserialized[i]);
}

TEST_F(SnapshotSerializerTest, VectorMt19937_64) {
  std::vector<std::mt19937_64> snapshot;
  for (int i = 123; i <= 321; i++)
    snapshot.emplace_back(i);

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<std::vector<std::mt19937_64>>(serialized);

  ASSERT_EQ(snapshot.size(), deserialized.size());
  for (size_t i = 0; i < snapshot.size(); i++)
    EXPECT_EQ(snapshot[i], deserialized[i]);
}

TEST_F(SnapshotSerializerTest, LoaderStateSnapshot) {
  LoaderStateSnapshot snapshot = {
    std::default_random_engine(123),
    321
  };

  std::string serialized = SnapshotSerializer().Serialize(snapshot);
  auto deserialized = SnapshotSerializer().Deserialize<LoaderStateSnapshot>(serialized);

  EXPECT_EQ(snapshot.rng, deserialized.rng);
  EXPECT_EQ(snapshot.current_epoch, deserialized.current_epoch);
}

}  // namespace dali
