// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef DALI_OPERATORS_IMAGE_REMAP_JITTER_CUH_
#define DALI_OPERATORS_IMAGE_REMAP_JITTER_CUH_

#include <ctgmath>
#include <vector>
#include <random>
#include <string>
#include "dali/core/host_dev.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/image/remap/displacement_filter.h"
#include "dali/operators/util/randomizer.cuh"
#include "dali/operators/random/rng_checkpointing_utils.h"

namespace dali {

template <typename Backend>
class JitterAugment {};

template <>
class JitterAugment<GPUBackend> {
 public:
  static constexpr bool is_stateless = false;
  explicit JitterAugment(const OpSpec& spec) :
        rnd_(spec.GetArgument<int64_t>("seed"), rnd_size_),
        nDegree_(spec.GetArgument<int>("nDegree")) {
  }

  __device__ ivec2 operator()(int y, int x, int c, int H, int W, int C) {
    const uint16_t nHalf = nDegree_/2;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int newX = curand(&rnd_[idx % rnd_size_]) % nDegree_ - nHalf + x;
    int newY = curand(&rnd_[idx % rnd_size_]) % nDegree_ - nHalf + y;
    return { cuda_min(cuda_max(0, newX), W), cuda_min(cuda_max(0, newY), H) };
  }

  void Cleanup() {}

  curand_states rnd_;

 private:
  int nDegree_;
  static constexpr unsigned rnd_size_ = 1024 * 256;
};

template <typename Backend>
class Jitter : public DisplacementFilter<Backend, JitterAugment<Backend>> {
 public:
  inline explicit Jitter(const OpSpec &spec)
    : DisplacementFilter<Backend, JitterAugment<Backend>>(spec) {}

  virtual ~Jitter() = default;

  using DisplacementFilter<Backend, JitterAugment<Backend>>::displace_;
  using CheckpointType = decltype(JitterAugment<Backend>::rnd_);
  using CheckpointUtils = rng::RngCheckpointUtils<GPUBackend, CheckpointType>;

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    static_assert(std::is_same_v<Backend, GPUBackend>);
    CheckpointUtils::SaveState(cpt, order, displace_.rnd_);
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    static_assert(std::is_same_v<Backend, GPUBackend>);
    CheckpointUtils::RestoreState(cpt, displace_.rnd_);
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    static_assert(std::is_same_v<Backend, GPUBackend>);
    return CheckpointUtils::SerializeCheckpoint(cpt);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    static_assert(std::is_same_v<Backend, GPUBackend>);
    CheckpointUtils::DeserializeCheckpoint(cpt, data);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_JITTER_CUH_
