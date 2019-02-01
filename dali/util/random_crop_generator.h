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

#ifndef DALI_UTIL_RANDOM_CROP_GENERATOR_H_
#define DALI_UTIL_RANDOM_CROP_GENERATOR_H_

#include <vector>
#include <random>
#include <utility>
#include "dali/common.h"
#include "dali/util/crop_window.h"

namespace dali {

using AspectRatioRange = std::pair<float, float>;
using AreaRange = std::pair<float, float>;

class DLL_PUBLIC RandomCropGenerator {
 public:
  RandomCropGenerator(AspectRatioRange aspect_ratio_range,
                      AreaRange area_range,
                      int64_t seed = time(0),
                      int num_attempts_ = 10);

  DLL_PUBLIC CropWindow GenerateCropWindow(int H, int W);
  DLL_PUBLIC std::vector<CropWindow> GenerateCropWindows(int H, int W, std::size_t N);

 private:
  CropWindow GenerateCropWindowImpl(int H, int W);

  std::uniform_real_distribution<float> aspect_ratio_dis_;
  std::uniform_real_distribution<float> area_dis_;
  std::uniform_real_distribution<float> uniform_;
  std::mt19937 rand_gen_;
  int64_t seed_;
  int num_attempts_;
};

}  // namespace dali

#endif  // DALI_UTIL_RANDOM_CROP_GENERATOR_H_
