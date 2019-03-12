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
#include "dali/util/crop_window.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class CropAttr {
 protected:
  explicit inline CropAttr(const OpSpec &spec)
    : spec__(spec)
    , batch_size__(spec__.GetArgument<int>("batch_size")) {
    int crop_h = 0, crop_w = 0;
    bool has_crop_arg = spec__.HasArgument("crop");
    bool has_crop_w_arg = spec__.HasArgument("crop_w");
    bool has_crop_h_arg = spec__.HasArgument("crop_h");
    if (has_crop_arg) {
      DALI_ENFORCE(!has_crop_h_arg && !has_crop_w_arg,
        "crop argument is not compatible with crop_h, crop_w");

      auto cropArg = spec.GetRepeatedArgument<float>("crop");
      DALI_ENFORCE(cropArg.size() > 0 && cropArg.size() <= 2);
      crop_h = static_cast<int>(cropArg[0]);
      crop_w = static_cast<int>(cropArg.size() == 2 ? cropArg[1] : cropArg[0]);

      DALI_ENFORCE(crop_h >= 0,
        "Crop height must be greater than zero. Received: " +
        std::to_string(crop_h));

      DALI_ENFORCE(crop_w >= 0,
        "Crop width must be greater than zero. Received: " +
        std::to_string(crop_w));
    } else if (has_crop_h_arg || has_crop_w_arg) {
      DALI_ENFORCE(has_crop_w_arg && has_crop_h_arg,
        "Both crop_w and crop_h arguments must be provided");
    }

    crop_height_.resize(batch_size__, crop_h);
    crop_width_.resize(batch_size__, crop_w);
    crop_x_norm_.resize(batch_size__, 0.0f);
    crop_y_norm_.resize(batch_size__, 0.0f);
    crop_window_generators_.resize(batch_size__, {});
  }

  void ProcessArguments(const ArgumentWorkspace *ws, std::size_t data_idx) {
    crop_x_norm_[data_idx] = spec__.GetArgument<float>("crop_pos_x", ws, data_idx);
    crop_y_norm_[data_idx] = spec__.GetArgument<float>("crop_pos_y", ws, data_idx);
    if (crop_width_[data_idx] == 0) {
      crop_width_[data_idx] =
        static_cast<int>(spec__.GetArgument<float>("crop_w", ws, data_idx));
    }
    if (crop_height_[data_idx] == 0) {
      crop_height_[data_idx] =
          static_cast<int>(spec__.GetArgument<float>("crop_h", ws, data_idx));
    }

    crop_window_generators_[data_idx] =
      [this, data_idx](int H, int W) {
        CropWindow crop_window;
        crop_window.h = crop_height_[data_idx];
        crop_window.w = crop_width_[data_idx];
        std::tie(crop_window.y, crop_window.x) =
          CalculateCropYX(
            crop_y_norm_[data_idx], crop_x_norm_[data_idx],
            crop_window.h, crop_window.w,
            H, W);
        DALI_ENFORCE(crop_window.IsInRange(H, W));
        return crop_window;
      };
  }

  void ProcessArguments(const ArgumentWorkspace *ws) {
    for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
      ProcessArguments(ws, data_idx);
    }
  }

  void ProcessArguments(const SampleWorkspace *ws) {
    ProcessArguments(ws, ws->data_idx());
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
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
    DALI_ENFORCE(crop_W > 0 && crop_W <= W, "Invalid crop_width: " + std::to_string(crop_W)
      + " (image_width: " + std::to_string(W) + ")");
    DALI_ENFORCE(crop_H > 0 && crop_H <= H, "Invalid crop_heigth: " + std::to_string(crop_H)
      + " (image_heigth: " + std::to_string(H) + ")");

    const int crop_y = crop_y_norm * (H - crop_H);
    const int crop_x = crop_x_norm * (W - crop_W);

    return std::make_pair(crop_y, crop_x);
  }

  std::vector<int> crop_height_;
  std::vector<int> crop_width_;
  std::vector<float> crop_x_norm_;
  std::vector<float> crop_y_norm_;
  std::vector<CropWindowGenerator> crop_window_generators_;

 private:
  OpSpec spec__;
  std::size_t batch_size__;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_ATTR_H_
