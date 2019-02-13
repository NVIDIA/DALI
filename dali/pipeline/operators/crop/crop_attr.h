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

#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_ATTR_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_ATTR_H_

#include <utility>
#include <vector>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class CropAttr {
 protected:
  explicit inline CropAttr(const OpSpec &spec) {
    if (spec.name() != "Resize") {
      const int batch_size = spec.GetArgument<int>("batch_size");
      vector<float> cropArgs = spec.GetRepeatedArgument<float>("crop");

      DALI_ENFORCE(cropArgs[0] >= 0,
        "Crop height must be greater than zero. Received: " +
        std::to_string(cropArgs[0]));

      DALI_ENFORCE(cropArgs[1] >= 0,
        "Crop width must be greater than zero. Received: " +
        std::to_string(cropArgs[1]));

      crop_height_ = std::vector<int>(batch_size, static_cast<int>(cropArgs[0]));
      crop_width_ = std::vector<int>(batch_size, static_cast<int>(cropArgs[1]));
    }
  }

  /**
   * @brief Calculate coordinate where the crop starts in pixels.
   */
  std::pair<int, int> CalculateCropYX(float crop_y_norm, float crop_x_norm,
                                      int crop_H, int crop_W,
                                      int H, int W) {
    DALI_ENFORCE(crop_y_norm >= 0.f && crop_y_norm <= 1.f,
                "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_norm >= 0.f && crop_x_norm <= 1.f,
                "Crop coordinates need to be in range [0.0, 1.0]");

    const int crop_y = crop_y_norm * (H - crop_H);
    const int crop_x = crop_x_norm * (W - crop_W);

    return std::make_pair(crop_y, crop_x);
  }

  /**
   * @brief Calculate coordinate where the crop starts in pixels.
   *
   * @param spec
   * @param ws
   * @param imgIdx
   * @param H
   * @param W
   * @return std::pair<int, int>
   */
  std::pair<int, int> CalculateCropYX(const OpSpec &spec,
                                      const ArgumentWorkspace *ws,
                                      const Index dataIdx, int H, int W) {
    auto crop_x_norm = spec.GetArgument<float>("crop_pos_x", ws, dataIdx);
    auto crop_y_norm = spec.GetArgument<float>("crop_pos_y", ws, dataIdx);
    return CalculateCropYX(
      crop_y_norm, crop_x_norm,
      crop_height_[dataIdx], crop_width_[dataIdx],
      H, W);
  }

  std::vector<int> crop_height_;
  std::vector<int> crop_width_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_ATTR_H_
