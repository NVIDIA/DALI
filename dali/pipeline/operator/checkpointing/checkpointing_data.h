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

#ifndef DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINTING_DATA_H_
#define DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINTING_DATA_H_

#include <random>
#include <variant>
#include <vector>

namespace dali {

/**
 * @brief A batch of CPU 32bit random number generators.
 *
 * Used by ImageRandomCrop operator.
*/
struct RNGSnapshotCPU {
  std::vector<std::mt19937> rng;
};

/**
 * @brief A batch of CPU 64bit random number generators.
 *
 * Used by CPU fn.random operators.
*/
struct RNGSnapshotCPU64 {
  std::vector<std::mt19937_64> rng;
};

/**
 * @brief Structure describing Loader base class state, at the begining of an epoch.
*/
struct LoaderStateSnapshot {
  std::default_random_engine rng;
  int current_epoch;
};

/**
 * @brief DataReader state.
 *
 * Keeps loader state. Made into a separate entity to allow extension.
*/
struct ReaderStateSnapshot {
  LoaderStateSnapshot loader_state;
};

/**
 * @brief Dummy snapshot, used for testing.
*/
struct DummySnapshot {
  std::vector<uint8_t> dummy_state;
};

/**
 * @brief Universal checkpoint data storage type
*/
using CheckpointingData = std::variant<
  std::monostate,
  RNGSnapshotCPU,
  RNGSnapshotCPU64,
  ReaderStateSnapshot,
  DummySnapshot
>;

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINTING_DATA_H_
