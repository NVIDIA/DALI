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
 protected:
  explicit inline RandomCropAttr(const OpSpec &spec) {
    int64_t seed = spec.GetArgument<int64_t>("seed");
    int num_attempts = spec.GetArgument<int>("num_attempts");

    std::vector<float> aspect_ratio;
    GetSingleOrRepeatedArg(spec, &aspect_ratio, "random_aspect_ratio", 2);

    std::vector<float> area;
    GetSingleOrRepeatedArg(spec, &area, "random_area", 2);

    std::shared_ptr<RandomCropGenerator> random_crop_generator(
      new RandomCropGenerator(
        {aspect_ratio[0], aspect_ratio[1]},
        {area[0], area[1]},
        seed,
        num_attempts));

    crop_window_generator_ = std::bind(
      &RandomCropGenerator::GenerateCropWindow, random_crop_generator,
      std::placeholders::_1, std::placeholders::_2);
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    return crop_window_generator_;
  }

 private:
  CropWindowGenerator crop_window_generator_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_RANDOM_CROP_ATTR_H_
