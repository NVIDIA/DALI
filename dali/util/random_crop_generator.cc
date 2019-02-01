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

#include <utility>
#include "dali/util/random_crop_generator.h"
#include "dali/error_handling.h"

namespace dali {

RandomCropGenerator::RandomCropGenerator(
    AspectRatioRange aspect_ratio_range,
    AreaRange area_range,
    int64_t seed,
    int num_attempts)
  : aspect_ratio_dis_(aspect_ratio_range.first, aspect_ratio_range.second)
  , area_dis_(area_range.first, area_range.second)
  , uniform_(0.0f, 1.0f)
  , rand_gen_(seed)
  , seed_(seed)
  , num_attempts_(num_attempts) {
}

CropWindow RandomCropGenerator::GenerateCropWindowImpl(int H, int W) {
    DALI_ENFORCE(H > 0);
    DALI_ENFORCE(W > 0);

    for (int attempt = 0; attempt < num_attempts_; attempt++) {
        float scale = area_dis_(rand_gen_);
        float ratio = aspect_ratio_dis_(rand_gen_);
        float swap  = uniform_(rand_gen_);

        size_t original_area = H * W;
        float target_area = scale * original_area;

        int w = static_cast<int>(
            std::roundf(sqrtf(target_area * ratio)));
        int h = static_cast<int>(
            std::roundf(sqrtf(target_area / ratio)));

        if (swap < 0.5f) {
           std::swap(w, h);
        }

        CropWindow crop;
        if (w > 0 && h > 0 && w <= W && h <= H) {
            float rand_x = uniform_(rand_gen_);
            float rand_y = uniform_(rand_gen_);

            crop.w = w;
            crop.h = h;
            crop.x = static_cast<int>(rand_x * (W - w));
            crop.y = static_cast<int>(rand_y * (H - h));
            return crop;
        }
    }

    // If kMaxAttempts were consumed, use default crop
    int min_dim = H < W ? H : W;
    CropWindow crop;
    crop.w = min_dim;
    crop.h = min_dim;
    crop.x = (W - min_dim) / 2;
    crop.y = (H - min_dim) / 2;
    return crop;
}

CropWindow RandomCropGenerator::GenerateCropWindow(int H, int W) {
    return GenerateCropWindowImpl(H, W);
}

std::vector<CropWindow> RandomCropGenerator::GenerateCropWindows(int H, int W, std::size_t N) {
    std::seed_seq seq{seed_};
    std::vector<int64_t> seeds(N);
    seq.generate(seeds.begin(), seeds.end());

    std::vector<CropWindow> crop_windows;
    for (std::size_t i = 0; i < N; i++) {
        rand_gen_.seed(seeds[i]);
        crop_windows.push_back(
            GenerateCropWindowImpl(H, W));
    }
    return crop_windows;
}

}  // namespace dali
