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

#ifndef DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_ATTR_H_
#define DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_ATTR_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"
#include "dali/operators/image/crop/random_crop_generator_util.h"
#include "dali/operators/random/rng_base.h"

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
    GetSingleOrRepeatedArg(spec, aspect_ratio, "random_aspect_ratio", 2);
    DALI_ENFORCE(aspect_ratio[0] <= aspect_ratio[1], "Provided empty range");

    std::vector<float> area;
    GetSingleOrRepeatedArg(spec, area, "random_area", 2);
    DALI_ENFORCE(area[0] <= area[1], "Provided empty range");

    int max_batch_size = spec.GetArgument<int>("max_batch_size");

    int64_t seed;
    if (!spec.TryGetArgument(seed, "seed"))
      seed = time(0);

    seed ^= 0x12345678abcdefe_u64;

    random_crop_generators_.reserve(max_batch_size);
    for (int i = 0; i < max_batch_size; i++) {
      random_crop_generators_.emplace_back(
          Philox4x32_10::State(seed, rng::kSkipaheadPerSample * i, 0),
          AspectRatioRange{aspect_ratio[0], aspect_ratio[1]},
          AreaRange{area[0], area[1]}, num_attempts);
    }
  }

  inline CropWindowGenerator GetCropWindowGenerator(int idx) {
    return [idx, this](const TensorShape<>& shape, const TensorLayout&) {
      return random_crop_generators_[idx].GenerateCropWindow(shape);
    };
  }

 protected:
  std::vector<RandomCropGenerator> random_crop_generators_;
};

template <typename Base>
class OperatorWithRandomCrop
: public rng::OperatorWithRng<Base>, public RandomCropAttr {
 protected:
  explicit OperatorWithRandomCrop(const OpSpec &spec) :
    rng::OperatorWithRng<Base>(spec), RandomCropAttr(spec) {}

  void LoadRandomState(const Workspace &ws) override {
    rng::OperatorWithRng<Base>::LoadRandomState(ws);

    for (size_t i = 0; i < random_crop_generators_.size(); i++) {
      auto state = this->GetSampleRNG(i).get_state();
      state.key ^= 0x12345678abcdefe_u64;
      random_crop_generators_[i].SetRNGState(state);
    }
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_ATTR_H_
