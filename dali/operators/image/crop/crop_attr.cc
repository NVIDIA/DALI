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

#include <vector>
#include "dali/operators/image/crop/crop_attr.h"

namespace dali {

DALI_SCHEMA(CropAttr)
    .DocStr(R"code(Crops attributes placeholder)code")
    .AddOptionalArg(
        "crop",
        R"code(Shape of the cropped image, specified as a list of value (e.g. `(crop_H, crop_W)` for
2D crop, `(crop_D, crop_H, crop_W)` for volumetric crop). Providing `crop` argument is incompatible
with providing separate arguments `crop_d`, `crop_h` and `crop_w`.)code",
        std::vector<float>{0.f, 0.f})
    .AddOptionalArg(
        "crop_pos_x",
        R"code(Normalized (0.0 - 1.0) horizontal position of the cropping window (upper left corner).
Actual position is calculated as `crop_x = crop_x_norm * (W - crop_W)`,
where `crop_x_norm` is the normalized position, `W` is the width of the image
and `crop_W` is the width of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_y",
        R"code(Normalized (0.0 - 1.0) vertical position of the cropping window (upper left corner).
Actual position is calculated as `crop_y = crop_y_norm * (H - crop_H)`,
where `crop_y_norm` is the normalized position, `H` is the height of the image
and `crop_H` is the height of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_z",
        R"code(**Volumetric inputs only** Normalized (0.0 - 1.0) normal position of the cropping window (front plane).
Actual position is calculated as `crop_z = crop_z_norm * (D - crop_d)`,
where `crop_z_norm` is the normalized position, `D` is the depth of the image
and `crop_d` is the depth of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_w",
        R"code(Cropping window width (in pixels).
If provided, `crop_h` should be provided as well. Providing `crop_w`, `crop_h` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_h",
        R"code(Cropping window height (in pixels).
If provided, `crop_w` should be provided as well. Providing `crop_w`, `crop_h` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_d",
        R"code(**Volumetric inputs only** cropping window depth (in pixels).
If provided, `crop_h` and `crop_w` should be provided as well. Providing `crop_w`, `crop_h`, `crop_d` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true);

}  // namespace dali
