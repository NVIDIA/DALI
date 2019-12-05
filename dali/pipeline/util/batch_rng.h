// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_BATCH_RNG_H_
#define DALI_PIPELINE_UTIL_BATCH_RNG_H_

#include <random>
#include <vector>

#include "dali/core/span.h"

namespace dali {

template <typename RNG = std::mt19937>
class BatchRNG {
 public:
  /**
   * @brief Used to keep batch of RNGs, so Operators can be immune to order of sample processing
   * while using randomness
   *
   * @param seed Used to generate seed_seq to initialize batch of RNGs
   * @param batch_size How many RNGs to store
   * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
   * collisions between seeds used to initialize RNGs in different operators.
   */
  BatchRNG(int64_t seed, int batch_size, int state_size = 4) : seed_(seed) {
    std::seed_seq seq{seed_};
    std::vector<uint32_t> seeds(batch_size * state_size);
    seq.generate(seeds.begin(), seeds.end());
    rngs_.reserve(batch_size);
    for (int i = 0; i < batch_size * state_size; i += state_size) {
      std::seed_seq s(seeds.begin() + i, seeds.begin() + i + state_size);
      rngs_.emplace_back(s);
    }
  }


  /**
   * Returns engine corresponding to given sample ID
   */
  RNG &operator[](int sample) noexcept {
    return rngs_[sample];
  }


 private:
  int64_t seed_;
  std::vector<RNG> rngs_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BATCH_RNG_H_
