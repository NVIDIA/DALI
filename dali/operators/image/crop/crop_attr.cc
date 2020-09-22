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
        "crop", R"code(Shape of the cropped image, specified as a list of values (for example,
``(crop_H, crop_W)`` for the 2D crop and ``(crop_D, crop_H, crop_W)`` for the volumetric crop).

Providing crop argument is incompatible with providing separate arguments such as ``crop_d``,
``crop_h``, and ``crop_w``.)code",
        std::vector<float>{0.f, 0.f})
    .AddOptionalArg(
        "crop_pos_x", R"code(Normalized (0.0 - 1.0) horizontal position of the cropping window
(upper left corner).

The actual position is calculated as ``crop_x = crop_x_norm * (W - crop_W)``, where `crop_x_norm`
is the normalized position, ``W`` is the width of the image, and ``crop_W`` is the width of the
cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
         "crop_pos_y", R"code(Normalized (0.0 - 1.0) vertical position of the start of
the cropping window (typically, the upper left corner).

The actual position is calculated as ``crop_y = crop_y_norm * (H - crop_H)``, where ``crop_y_norm``
is the normalized position, `H` is the height of the image, and ``crop_H`` is the
height of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_pos_z", R"code(Applies **only** to volumetric inputs.

Normalized (0.0 - 1.0) normal position of the cropping window (front plane).
The actual position is calculated as ``crop_z = crop_z_norm * (D - crop_D)``, where ``crop_z_norm``
is the normalized position, ``D`` is the depth of the image and ``crop_D`` is the depth
of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg(
        "crop_w", R"code(Cropping window width (in pixels).

Providing values for ``crop_w`` and ``crop_h`` is incompatible with providing fixed crop window
dimensions (argument ``crop``).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_h", R"code(Cropping the window height (in pixels).

Providing values for ``crop_w`` and ``crop_h`` is incompatible with providing fixed crop
window dimensions (argument ``crop``).)code",
        0.0f, true)
    .AddOptionalArg(
        "crop_d", R"code(Applies **only** to volumetric inputs; cropping window depth (in voxels).

``crop_w``, ``crop_h``, and ``crop_d`` must be specified together. Providing values
for ``crop_w``, ``crop_h``, and ``crop_d`` is incompatible with providing the fixed crop
window dimensions (argument `crop`).)code",
        0.0f, true);

}  // namespace dali
