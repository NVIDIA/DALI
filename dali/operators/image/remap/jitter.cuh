// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/random/rng_util.cuh"
#include "dali/operators/random/rng_base.h"
#include "dali/operators/random/random_dist.h"

namespace dali {

template <typename Backend>
class JitterAugment {};

template <>
class JitterAugment<GPUBackend> {
 public:
  static constexpr bool is_stateless = false;
  explicit JitterAugment(const OpSpec& spec) : nDegree_(spec.GetArgument<int>("nDegree")) {}

  __device__ ivec2 operator()(int sample_idx, int y, int x, int c, int H, int W, int C) const {
    const int center = -(nDegree_ >> 1);
    auto state = rng_state_;
    auto rnd = ToCurand(state);
    skipahead_sequence(sample_idx * rng::kSkipaheadPerSample, &rnd);
    ptrdiff_t offset = c + C * (x + W * y);  // component offset in HWC layout
    skipahead(offset * rng::kSkipaheadPerElement, &rnd);

    auto dist = random::uniform_int_dist<int>(center, center + nDegree_, true);
    auto gen = random::CurandGenerator(rnd);
    int newX = dist(gen) + x;
    int newY = dist(gen) + y;
    return { clamp(newX, 0, W - 1), clamp(newY, 0, H - 1) };
  }

  void Cleanup() {}

  Philox4x32_10::State rng_state_;

 private:
  int nDegree_;
};

template <typename Backend>
class Jitter : public rng::OperatorWithRng<DisplacementFilter<Backend, JitterAugment<Backend>>> {
 public:
  using Base = rng::OperatorWithRng<DisplacementFilter<Backend, JitterAugment<Backend>>>;
  inline explicit Jitter(const OpSpec &spec)
    : Base(spec) {}

  void RunImpl(Workspace &ws) override {
    displace_.rng_state_ = this->GetSampleRNG(0).get_state();
    Base::RunImpl(ws);
  }

  using Base::displace_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_JITTER_CUH_
