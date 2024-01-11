// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_RNG_CHECKPOINTING_UTILS_H_
#define DALI_OPERATORS_RANDOM_RNG_CHECKPOINTING_UTILS_H_

#include <random>
#include <string>
#include <vector>
#include <memory>

#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/operators/util/randomizer.cuh"

namespace dali {
namespace rng {

template <typename Backend, typename Rng>
class RngCheckpointUtils;

template <typename Rng>
class RngCheckpointUtils<CPUBackend, BatchRNG<Rng>> {
 public:
  static void SaveState(OpCheckpoint &cpt, AccessOrder order, const BatchRNG<Rng> &rng) {
    cpt.MutableCheckpointState() = rng;
  }

  static void RestoreState(const OpCheckpoint &cpt, BatchRNG<Rng> &rng) {
    const auto &restored = cpt.CheckpointState<BatchRNG<Rng>>();
    DALI_ENFORCE(restored.BatchSize() == rng.BatchSize(),
                  "Provided checkpoint doesn't match the expected batch size. "
                  "Perhaps the batch size setting changed? ");
    rng = restored;
  }

  static std::string SerializeCheckpoint(const OpCheckpoint &cpt) {
    const auto &state = cpt.CheckpointState<BatchRNG<Rng>>();
    return SnapshotSerializer().Serialize(state.ToVector());
  }

  static void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) {
    auto deserialized = SnapshotSerializer().Deserialize<std::vector<Rng>>(data);
    cpt.MutableCheckpointState() = BatchRNG<Rng>::FromVector(deserialized);
  }
};


template <>
class RngCheckpointUtils<GPUBackend, curand_states> {
 public:
  static void SaveState(OpCheckpoint &cpt, AccessOrder order, const curand_states &rng) {
    cpt.SetOrder(order);
    cpt.MutableCheckpointState() = rng.copy(order);
    // The pipeline will perform host synchronization before serializing the checkpoints.
    // TODO(skarpinski) Move synchronization out from pipeline's GetCheckpoint.
  }

  static void RestoreState(const OpCheckpoint &cpt, curand_states &rng) {
    const auto &states_gpu = cpt.CheckpointState<curand_states>();
    DALI_ENFORCE(states_gpu.length() == rng.length(),
                "Provided checkpoint doesn't match the expected batch size. "
                "Perhaps the batch size setting changed? ");
    rng.set(states_gpu);
  }

  static std::string SerializeCheckpoint(const OpCheckpoint &cpt) {
    const auto &states_gpu = cpt.CheckpointState<curand_states>();
    size_t n = states_gpu.length();
    std::vector<curandState> states(n);
    CUDA_CALL(cudaMemcpy(states.data(), states_gpu.states(), n * sizeof(curandState),
                         cudaMemcpyDeviceToHost));
    return SnapshotSerializer().Serialize(states);
  }

  static void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) {
    auto deserialized = SnapshotSerializer().Deserialize<std::vector<curandState>>(data);
    curand_states states(deserialized.size());
    CUDA_CALL(cudaMemcpyAsync(states.states(),
                              deserialized.data(),
                              sizeof(curandState) * deserialized.size(),
                              cudaMemcpyHostToDevice, cudaStreamDefault));
    CUDA_CALL(cudaStreamSynchronize(cudaStreamDefault));
    cpt.MutableCheckpointState() = states;
  }
};

}  // namespace rng
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_CHECKPOINTING_UTILS_H_
