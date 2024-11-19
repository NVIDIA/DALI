// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/crop/crop_attr.h"
#include <cmath>
#include <string>

namespace dali {

DALI_SCHEMA(CropAttr)
    .DocStr(R"code(Crops attributes placeholder)code")
    .AddOptionalArg<std::vector<float>>(
        "crop", R"code(Shape of the cropped image, specified as a list of values (for example,
``(crop_H, crop_W)`` for the 2D crop and ``(crop_D, crop_H, crop_W)`` for the volumetric crop).

Providing crop argument is incompatible with providing separate arguments such as `crop_d`,
`crop_h`, and `crop_w`.)code",
        nullptr, true)
    .AddOptionalArg(
        "crop_pos_x", R"code(Normalized (0.0 - 1.0) horizontal position of the cropping window
(upper left corner).

The actual position is calculated as ``crop_x = crop_x_norm * (W - crop_W)``, where `crop_x_norm`
is the normalized position, ``W`` is the width of the image, and ``crop_W`` is the width of the
cropping window.

See `rounding` argument for more details on how ``crop_x`` is converted to an integral value.)code",
        0.5f, true)
    .AddOptionalArg(
         "crop_pos_y", R"code(Normalized (0.0 - 1.0) vertical position of the start of
the cropping window (typically, the upper left corner).

The actual position is calculated as ``crop_y = crop_y_norm * (H - crop_H)``, where ``crop_y_norm``
is the normalized position, `H` is the height of the image, and ``crop_H`` is the height of the
cropping window.

See `rounding` argument for more details on how ``crop_y`` is converted to an integral value.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_z", R"code(Applies **only** to volumetric inputs.

Normalized (0.0 - 1.0) normal position of the cropping window (front plane).
The actual position is calculated as ``crop_z = crop_z_norm * (D - crop_D)``, where ``crop_z_norm``
is the normalized position, ``D`` is the depth of the image and ``crop_D`` is the depth of the
cropping window.

See `rounding` argument for more details on how ``crop_z`` is converted to an integral value.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_w", R"code(Cropping window width (in pixels).

Providing values for `crop_w` and `crop_h` is incompatible with providing fixed crop window
dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_h", R"code(Cropping the window height (in pixels).

Providing values for `crop_w` and `crop_h` is incompatible with providing fixed crop
window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_d", R"code(Applies **only** to volumetric inputs; cropping window depth (in voxels).

`crop_w`, `crop_h`, and `crop_d` must be specified together. Providing values
for `crop_w`, `crop_h`, and `crop_d` is incompatible with providing the fixed crop
window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg(
        "rounding", R"code(Determines the rounding function used to convert the starting coordinate
of the window to an integral value (see `crop_pos_x`, `crop_pos_y`, `crop_pos_z`).

Possible values are:

* | ``"round"`` - Rounds to the nearest integer value, with halfway cases rounded away from zero.
* | ``"truncate"`` - Discards the fractional part of the number (truncates towards zero).)code",
        "round");

CropAttr::CropAttr(const OpSpec& spec) {
  auto max_batch_size = spec.GetArgument<int>("max_batch_size");
  bool has_crop_arg = spec.ArgumentDefined("crop");
  bool has_crop_w_arg = spec.ArgumentDefined("crop_w");
  bool has_crop_h_arg = spec.ArgumentDefined("crop_h");
  bool has_crop_d_arg = spec.ArgumentDefined("crop_d");
  is_whole_image_ = !has_crop_arg && !has_crop_w_arg && !has_crop_h_arg && !has_crop_d_arg;

  DALI_ENFORCE(has_crop_w_arg == has_crop_h_arg,
               "`crop_w` and `crop_h` arguments must be provided together");

  if (has_crop_d_arg) {
    DALI_ENFORCE(has_crop_w_arg,
                 "`crop_d` argument must be provided together with `crop_w` and `crop_h`");
  }

  int crop_d = kNoCrop, crop_h = kNoCrop, crop_w = kNoCrop;
  if (has_crop_arg) {
    DALI_ENFORCE(!has_crop_h_arg && !has_crop_w_arg && !has_crop_d_arg,
                 "`crop` argument is not compatible with `crop_h`, `crop_w`, `crop_d`");
    if (spec.HasArgument("crop")) {
      auto crop = spec.GetRepeatedArgument<float>("crop");
      DALI_ENFORCE(crop.size() >= 2 && crop.size() <= 3,
                   "`crop` argument should have 2 or 3 elements depending on the input data shape");
      int i = 0;
      if (crop.size() == 3) {
        crop_d = crop[i++];
      }
      crop_h = crop[i++];
      crop_w = crop[i++];
    }
  }
  crop_height_.resize(max_batch_size, crop_h);
  crop_width_.resize(max_batch_size, crop_w);
  crop_depth_.resize(max_batch_size, crop_d);
  crop_x_norm_.resize(max_batch_size, 0.0f);
  crop_y_norm_.resize(max_batch_size, 0.0f);
  crop_z_norm_.resize(max_batch_size, 0.0f);
  crop_window_generators_.resize(max_batch_size, {});

  auto rounding = spec.GetArgument<std::string>("rounding");
  if (rounding == "round") {
    round_fn_ = [](double x) {
      return static_cast<int64_t>(std::round(x));
    };
  } else if (rounding == "truncate") {
    round_fn_ = [](double x) {
      return static_cast<int64_t>(x);
    };
  } else {
    DALI_FAIL(make_string("``rounding`` value ", rounding,
                          " is not supported. Supported values are \"round\", or \"truncate\"."));
  }
}

void CropAttr::ProcessArguments(const OpSpec& spec, const ArgumentWorkspace* ws,
                                std::size_t data_idx) {
  int crop_arg_len = 0;
  if (spec.HasTensorArgument("crop")) {
    auto crop_arg = view<const float, 1>(ws->ArgumentInput("crop"))[data_idx];
    crop_arg_len = crop_arg.shape[0];
    DALI_ENFORCE(crop_arg_len >= 2 && crop_arg_len <= 3,
                 "`crop` argument should have 2 or 3 elements depending on the input data shape");
    int idx = 0;
    if (crop_arg_len == 3) {
      crop_depth_[data_idx] = static_cast<int>(crop_arg.data[idx++]);
    }
    crop_height_[data_idx] = static_cast<int>(crop_arg.data[idx++]);
    crop_width_[data_idx] = static_cast<int>(crop_arg.data[idx++]);
  }

  if (spec.ArgumentDefined("crop_w")) {
    crop_width_[data_idx] = static_cast<int>(spec.GetArgument<float>("crop_w", ws, data_idx));
  }
  if (spec.ArgumentDefined("crop_h")) {
    crop_height_[data_idx] = static_cast<int>(spec.GetArgument<float>("crop_h", ws, data_idx));
  }
  if (spec.ArgumentDefined("crop_d")) {
    crop_depth_[data_idx] = static_cast<int>(spec.GetArgument<float>("crop_d", ws, data_idx));
  }

  crop_x_norm_[data_idx] = spec.GetArgument<float>("crop_pos_x", ws, data_idx);
  crop_y_norm_[data_idx] = spec.GetArgument<float>("crop_pos_y", ws, data_idx);
  if (spec.ArgumentDefined("crop_d") || crop_depth_[data_idx] != kNoCrop) {
    crop_z_norm_[data_idx] = spec.GetArgument<float>("crop_pos_z", ws, data_idx);
  }

  crop_window_generators_[data_idx] = [this, data_idx](const TensorShape<>& input_shape,
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

    if (crop_depth_[data_idx] > 0) {
      if (d_dim >= 0) {
        crop_shape[d_dim] = crop_depth_[data_idx];
        anchor_norm[d_dim] = crop_z_norm_[data_idx];
      } else if (d_dim < 0 && f_dim >= 0) {
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

TensorShape<> CropAttr::CalculateAnchor(const span<float>& anchor_norm,
                                        const TensorShape<>& crop_shape,
                                        const TensorShape<>& input_shape) {
  DALI_ENFORCE(anchor_norm.size() == crop_shape.size() && anchor_norm.size() == input_shape.size());

  TensorShape<> anchor;
  anchor.resize(anchor_norm.size());
  for (int dim = 0; dim < anchor_norm.size(); dim++) {
    DALI_ENFORCE(anchor_norm[dim] >= 0.0f && anchor_norm[dim] <= 1.0f,
                 "Anchor for dimension " + std::to_string(dim) + " (" +
                     std::to_string(anchor_norm[dim]) + ") is out of range [0.0, 1.0]");
    auto anchor_f = static_cast<double>(anchor_norm[dim]) * (input_shape[dim] - crop_shape[dim]);
    anchor[dim] = round_fn_(anchor_f);
  }

  return anchor;
}

void CropAttr::ProcessArguments(const OpSpec& spec, const Workspace& ws) {
  int batch_size = ws.GetInputBatchSize(0);
  for (int data_idx = 0; data_idx < batch_size; data_idx++) {
    ProcessArguments(spec, &ws, data_idx);
  }
}

void CropAttr::ProcessArguments(const OpSpec& spec, const SampleWorkspace& ws) {
  ProcessArguments(spec, &ws, ws.data_idx());
}

const CropWindowGenerator& CropAttr::GetCropWindowGenerator(std::size_t data_idx) const {
  DALI_ENFORCE(data_idx < crop_window_generators_.size());
  return crop_window_generators_[data_idx];
}

}  // namespace dali
