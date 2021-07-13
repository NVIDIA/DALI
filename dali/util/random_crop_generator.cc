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

#include <cmath>
#include <utility>
#include "dali/util/random_crop_generator.h"
#include "dali/core/error_handling.h"

namespace dali {

RandomCropGenerator::RandomCropGenerator(
    AspectRatioRange aspect_ratio_range,
    AreaRange area_range,
    int64_t seed,
    int num_attempts)
  : aspect_ratio_range_(aspect_ratio_range)
  , aspect_ratio_log_dis_(std::log(aspect_ratio_range.first), std::log(aspect_ratio_range.second))
  , area_dis_(area_range.first, area_range.second)
  , rand_gen_(seed)
  , seed_(seed)
  , num_attempts_(num_attempts) {
}

CropWindow RandomCropGenerator::GenerateCropWindowImpl(const TensorShape<>& shape) {
  assert(shape.size() == 2);
  CropWindow crop;
  int H = shape[0], W = shape[1];
  if (W <= 0 || H <= 0) {
    return crop;
  }

  float min_wh_ratio = aspect_ratio_range_.first;
  float max_wh_ratio = aspect_ratio_range_.second;
  float max_hw_ratio = 1 / aspect_ratio_range_.first;
  float min_area = W * H * area_dis_.a();
  int maxW = std::max<int>(1, H * max_wh_ratio);
  int maxH = std::max<int>(1, W * max_hw_ratio);

  // detect two impossible cases early
  if (H * maxW < min_area) {  // image too wide
    crop.SetShape({H, maxW});
  } else if (W * maxH < min_area) {  // image too tall
    crop.SetShape({maxH, W});
  } else {
    // it can still fail for very small images when size granularity matters
    int attempts_left = num_attempts_;
    for (; attempts_left > 0; attempts_left--) {
      float scale = area_dis_(rand_gen_);

      size_t original_area = H * W;
      float target_area = scale * original_area;

      float ratio = std::exp(aspect_ratio_log_dis_(rand_gen_));
      auto w = static_cast<int>(
          std::roundf(sqrtf(target_area * ratio)));
      auto h = static_cast<int>(
          std::roundf(sqrtf(target_area / ratio)));

      if (w < 1)
        w = 1;
      if (h < 1)
        h = 1;
      crop.SetShape({h, w});

      ratio = static_cast<float>(w) / h;
      if (w <= W && h <= H && ratio >= min_wh_ratio && ratio <= max_wh_ratio)
        break;
    }

    if (attempts_left <= 0) {
      float max_area = area_dis_.b() * W * H;
      float ratio = static_cast<float>(W)/H;
      if (ratio > max_wh_ratio) {
        crop.SetShape({H, maxW});
      } else if (ratio < min_wh_ratio) {
        crop.SetShape({maxH, W});
      } else {
        crop.SetShape({H, W});
      }
      float scale = std::min(1.0f, max_area / (crop.shape[0] * crop.shape[1]));
      crop.shape[0] = std::max<int>(1, crop.shape[0] * std::sqrt(scale));
      crop.shape[1] = std::max<int>(1, crop.shape[1] * std::sqrt(scale));
    }
  }

  crop.anchor[0] = std::uniform_int_distribution<int>(0, H - crop.shape[0])(rand_gen_);
  crop.anchor[1] = std::uniform_int_distribution<int>(0, W - crop.shape[1])(rand_gen_);
  return crop;
}

CropWindow RandomCropGenerator::GenerateCropWindow(const TensorShape<>& shape) {
    return GenerateCropWindowImpl(shape);
}

std::vector<CropWindow>
RandomCropGenerator::GenerateCropWindows(const TensorShape<>& shape,
                                         std::size_t N) {
  std::seed_seq seq{seed_};
  std::vector<int64_t> seeds(N);
  seq.generate(seeds.begin(), seeds.end());

  std::vector<CropWindow> crop_windows;
  for (std::size_t i = 0; i < N; i++) {
    rand_gen_.seed(seeds[i]);
    crop_windows.push_back(GenerateCropWindowImpl(shape));
  }
  return crop_windows;
}

}  // namespace dali
