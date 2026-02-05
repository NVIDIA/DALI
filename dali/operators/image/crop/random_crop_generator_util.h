// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_UTIL_H_
#define DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_UTIL_H_

#include <chrono>
#include <vector>
#include <random>
#include <utility>
#include "dali/core/common.h"
#include "dali/util/crop_window.h"
#include "dali/operators/random/philox.h"

namespace dali {

using AspectRatioRange = std::pair<float, float>;
using AreaRange = std::pair<float, float>;

class DLL_PUBLIC RandomCropGenerator {
 public:
  static inline Philox4x32_10::State DefaultRngState() {
    uint64_t key = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return { key, 0, 0 };
  }

  explicit DLL_PUBLIC RandomCropGenerator(
    AspectRatioRange aspect_ratio_range = { 3.0f/4, 4.0f/3 },
    AreaRange area_range = { 0.08, 1 },
    Philox4x32_10::State rng_state = DefaultRngState(),
    int num_attempts_ = 10);

  DLL_PUBLIC inline void SetRNGState(const Philox4x32_10::State &s) { rand_gen_.set_state(s); }
  DLL_PUBLIC CropWindow GenerateCropWindow(const TensorShape<>& shape);

 private:
  CropWindow GenerateCropWindowImpl(const TensorShape<>& shape);

  AspectRatioRange aspect_ratio_range_;
  // Aspect ratios are uniformly distributed on logarithmic scale.
  // This provides natural symmetry and smoothness of the distribution.
  std::uniform_real_distribution<float> aspect_ratio_log_dis_;
  std::uniform_real_distribution<float> area_dis_;
  Philox4x32_10 rand_gen_;
  int num_attempts_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_UTIL_H_
