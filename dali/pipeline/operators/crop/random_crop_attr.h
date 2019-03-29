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

#ifndef DALI_PIPELINE_OPERATORS_CROP_RANDOM_CROP_ATTR_H_
#define DALI_PIPELINE_OPERATORS_CROP_RANDOM_CROP_ATTR_H_

#include <utility>
#include <functional>
#include <memory>
#include <vector>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/util/crop_window.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class RandomCropAttr {
 public:
  explicit inline RandomCropAttr(const OpSpec &spec) {
    int num_attempts = spec.GetArgument<int>("num_attempts");

    std::vector<float> aspect_ratio;
    GetSingleOrRepeatedArg(spec, &aspect_ratio, "random_aspect_ratio", 2);
    DALI_ENFORCE(aspect_ratio[0] <= aspect_ratio[1], "Provided empty range");

    std::vector<float> area;
    GetSingleOrRepeatedArg(spec, &area, "random_area", 2);
    DALI_ENFORCE(area[0] <= area[1], "Provided empty range");

    int64_t seed = spec.GetArgument<int64_t>("seed");
    std::seed_seq seq{seed};
    int batch_size = spec.GetArgument<int>("batch_size");
    DALI_ENFORCE(batch_size > 0, "batch_size should be greater than 0");
    std::vector<int> seeds(batch_size);
    seq.generate(seeds.begin(), seeds.end());

    crop_window_generators_.resize(batch_size);

    for (int i = 0; i < batch_size; i++) {
      std::shared_ptr<RandomCropGenerator> random_crop_generator(
        new RandomCropGenerator(
          {aspect_ratio[0], aspect_ratio[1]}, {area[0], area[1]}, seeds[i], num_attempts));
      crop_window_generators_[i] = std::bind(
        &RandomCropGenerator::GenerateCropWindow, random_crop_generator,
        std::placeholders::_1, std::placeholders::_2);
    }
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    return crop_window_generators_[data_idx];
  }

 private:
  std::vector<CropWindowGenerator> crop_window_generators_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_RANDOM_CROP_ATTR_H_
