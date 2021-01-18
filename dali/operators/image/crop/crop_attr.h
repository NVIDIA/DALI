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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_

#include <cmath>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class CropAttr {
 public:
  static constexpr int kNoCrop = -1;
  explicit inline CropAttr(const OpSpec& spec) : spec__(spec) {
    auto max_batch_size = spec__.GetArgument<int>("max_batch_size");
    int crop_h = kNoCrop, crop_w = kNoCrop, crop_d = kNoCrop;
    bool has_crop_arg = spec__.HasArgument("crop");
    bool has_crop_w_arg = spec__.ArgumentDefined("crop_w");
    bool has_crop_h_arg = spec__.ArgumentDefined("crop_h");
    bool has_crop_d_arg = spec__.ArgumentDefined("crop_d");
    is_whole_image_ = !has_crop_arg && !has_crop_w_arg && !has_crop_h_arg && !has_crop_d_arg;

    DALI_ENFORCE(has_crop_w_arg == has_crop_h_arg,
                 "`crop_w` and `crop_h` arguments must be provided together");

    if (has_crop_d_arg) {
      DALI_ENFORCE(has_crop_w_arg,
        "`crop_d` argument must be provided together with `crop_w` and `crop_h`");
    }

    size_t crop_arg_ndims = 0;
    if (has_crop_arg) {
      DALI_ENFORCE(!has_crop_h_arg && !has_crop_w_arg && !has_crop_d_arg,
        "`crop` argument is not compatible with `crop_h`, `crop_w`, `crop_d`");

      auto cropArg = spec.GetRepeatedArgument<float>("crop");
      crop_arg_ndims = cropArg.size();
      DALI_ENFORCE(crop_arg_ndims >= 2 && crop_arg_ndims <= 3,
        "`crop` argument should have 2 or 3 elements depending on the input data shape");

      size_t idx = 0;
      if (crop_arg_ndims == 3) {
        crop_d = static_cast<int>(cropArg[idx++]);
      }
      crop_h = static_cast<int>(cropArg[idx++]);
      crop_w = static_cast<int>(cropArg[idx++]);
    }
    has_crop_d_ = has_crop_d_arg || crop_arg_ndims == 3;

    crop_height_.resize(max_batch_size, crop_h);
    crop_width_.resize(max_batch_size, crop_w);
    if (has_crop_d_)
      crop_depth_.resize(max_batch_size, crop_d);
    crop_x_norm_.resize(max_batch_size, 0.0f);
    crop_y_norm_.resize(max_batch_size, 0.0f);
    if (has_crop_d_)
      crop_z_norm_.resize(max_batch_size, 0.0f);
    crop_window_generators_.resize(max_batch_size, {});
  }

  void ProcessArguments(const ArgumentWorkspace *ws, std::size_t data_idx) {
    crop_x_norm_[data_idx] = spec__.GetArgument<float>("crop_pos_x", ws, data_idx);
    crop_y_norm_[data_idx] = spec__.GetArgument<float>("crop_pos_y", ws, data_idx);
    if (has_crop_d_)
      crop_z_norm_[data_idx] = spec__.GetArgument<float>("crop_pos_z", ws, data_idx);
    if (spec__.ArgumentDefined("crop_w")) {
      crop_width_[data_idx] = static_cast<int>(
        spec__.GetArgument<float>("crop_w", ws, data_idx));
    }
    if (spec__.ArgumentDefined("crop_h")) {
      crop_height_[data_idx] = static_cast<int>(
        spec__.GetArgument<float>("crop_h", ws, data_idx));
    }
    if (spec__.ArgumentDefined("crop_d")) {
      crop_depth_[data_idx] = static_cast<int>(
        spec__.GetArgument<float>("crop_d", ws, data_idx));
    }

    crop_window_generators_[data_idx] =
      [this, data_idx](const TensorShape<>& input_shape,
                       const TensorLayout& shape_layout) {
        DALI_ENFORCE(input_shape.size() == shape_layout.size());
        CropWindow crop_window;
        auto crop_shape = input_shape;

        auto ndim = input_shape.size();
        int d_dim = shape_layout.find('D');
        int f_dim = shape_layout.find('F');
        int h_dim = shape_layout.find('H');
        int w_dim = shape_layout.find('W');

        DALI_ENFORCE(h_dim >= 0 && w_dim >= 0,
          "[H]eight and [W]idth must be present in the layout. Got: " + shape_layout.str());

        SmallVector<float, 4> anchor_norm;
        anchor_norm.resize(ndim, 0.5f);

        if (h_dim >= 0 && crop_height_[data_idx] > 0) {
          crop_shape[h_dim] = crop_height_[data_idx];
          anchor_norm[h_dim] = crop_y_norm_[data_idx];
        }

        if (w_dim >= 0 && crop_width_[data_idx] > 0) {
          crop_shape[w_dim] = crop_width_[data_idx];
          anchor_norm[w_dim] = crop_x_norm_[data_idx];
        }

        if (has_crop_d_) {
          if (d_dim >= 0 && crop_depth_[data_idx] > 0) {
            crop_shape[d_dim] = crop_depth_[data_idx];
            anchor_norm[d_dim] = crop_z_norm_[data_idx];
          } else if (d_dim < 0 && f_dim >= 0 && crop_depth_[data_idx] > 0) {
            // Special case.
            // This allows using crop_d to crop on the sequence dimension,
            // by treating video inputs as a volume instead of a sequence
            crop_shape[f_dim] = crop_depth_[data_idx];
            anchor_norm[f_dim] = crop_z_norm_[data_idx];
          }
        }

        crop_window.SetAnchor(CalculateAnchor(make_span(anchor_norm), crop_shape, input_shape));
        crop_window.SetShape(crop_shape);
        return crop_window;
    };
  }

  TensorShape<> CalculateAnchor(const span<float>& anchor_norm,
                                const TensorShape<>& crop_shape,
                                const TensorShape<>& input_shape) {
    DALI_ENFORCE(anchor_norm.size() == crop_shape.size()
              && anchor_norm.size() == input_shape.size());

    TensorShape<> anchor;
    anchor.resize(anchor_norm.size());
    for (int dim = 0; dim < anchor_norm.size(); dim++) {
      DALI_ENFORCE(anchor_norm[dim] >= 0.0f && anchor_norm[dim] <= 1.0f,
        "Anchor for dimension " + std::to_string(dim) + " (" + std::to_string(anchor_norm[dim]) +
        ") is out of range [0.0, 1.0]");
      anchor[dim] = std::roundf(anchor_norm[dim] * (input_shape[dim] - crop_shape[dim]));
    }

    return anchor;
  }

  void ProcessArguments(const ArgumentWorkspace &ws) {
    auto max_batch_size = static_cast<size_t>(spec__.GetArgument<int>("max_batch_size"));
    for (std::size_t data_idx = 0; data_idx < max_batch_size; data_idx++) {
      ProcessArguments(&ws, data_idx);
    }
  }

  void ProcessArguments(const SampleWorkspace &ws) {
    ProcessArguments(&ws, ws.data_idx());
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

  inline bool IsWholeImage() const {
    return is_whole_image_;
  }

  std::vector<int> crop_height_;
  std::vector<int> crop_width_;
  std::vector<int> crop_depth_;
  std::vector<float> crop_x_norm_;
  std::vector<float> crop_y_norm_;
  std::vector<float> crop_z_norm_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  bool is_whole_image_ = false;
  bool has_crop_d_ = false;

 private:
  OpSpec spec__;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_
