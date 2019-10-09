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
#include "dali/pipeline/operators/crop/crop_attr.h"

namespace dali {

DALI_SCHEMA(CropAttr)
    .DocStr(R"code(Crops attributes placeholder)code")
    .AddOptionalArg("crop",
        R"code(Shape of the cropped image, specified as a list of values. Arguments `dims` and `dim_names` can
be used to specify the order of the given dimensions (see `dims` and `dim_names` description).
Providing `crop` argument is incompatible with providing separate arguments `crop_d`, `crop_h` and `crop_w`.)code",
        std::vector<float>{0.f, 0.f})
    .AddOptionalArg("crop_pos_x",
        R"code(Normalized (0.0 - 1.0) horizontal position of the cropping window (upper left corner).
Actual position is calculated as `crop_x = crop_x_norm * (W - crop_W)`,
where `crop_x_norm` is the normalized position, `W` is the width of the image
and `crop_W` is the width of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg("crop_pos_y",
        R"code(Normalized (0.0 - 1.0) vertical position of the cropping window (upper left corner).
Actual position is calculated as `crop_y = crop_y_norm * (H - crop_H)`,
where `crop_y_norm` is the normalized position, `H` is the height of the image
and `crop_H` is the height of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg("crop_pos_z",
        R"code(**Volumetric inputs only** Normalized (0.0 - 1.0) normal position of the cropping window (front plane).
Actual position is calculated as `crop_z = crop_z_norm * (D - crop_d)`,
where `crop_z_norm` is the normalized position, `D` is the depth of the image
and `crop_d` is the depth of the cropping window.)code",
        0.5f, true)
    .AddOptionalArg("crop_w",
        R"code(Cropping window width (in pixels).
If provided, `crop_h` should be provided as well. Providing `crop_w`, `crop_h` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg("crop_h",
        R"code(Cropping window height (in pixels).
If provided, `crop_w` should be provided as well. Providing `crop_w`, `crop_h` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg("crop_d",
        R"code(**Volumetric inputs only** cropping window depth (in pixels).
If provided, `crop_h` and `crop_w` should be provided as well. Providing `crop_w`, `crop_h`, `crop_d` is incompatible with
providing fixed crop window dimensions (argument `crop`).)code",
        0.0f, true)
    .AddOptionalArg("normalized_shape",
        R"code(If set to `true`, crop shape arguments are meant normalized (range [0.0, 1.0]) and
relative to the image dimensions:
`crop_shape[d] = crop_anchor_norm[d] * input_shape[d]`.
If set to `false`, crop shape arguments are expressed in absolute terms (number of steps))code",
        false)
    .AddOptionalArg("normalized_anchor",
        R"code(If set to `true`, crop anchor arguments are normalized (range [0.0, 1.0]) and
adjusted to represent the normalized position (from upper left corner of the data) as:
`crop_anchor[d] = crop_anchor_norm[d] * (input_shape[d] - crop_shape[d])`
If set to `false`, crop anchor arguments are expressed in absolute terms (number of steps for the
upper left corner))code",
        true)
    .AddOptionalArg("dims",
        R"code(If provided, it specifies the indexes of the dimensions that `crop` argument represents.
Example 1: `crop=(400,200,10)` and `dims=(2,1,0)` is equivalent to `crop_d=10`, `crop_h=200`, and `crop_w=400` (assuming "HWC" layout).
Example 2: `crop=(400,200)` and `dims=(1,0)` is equivalent to `crop_h=200`, and `crop_w=400` (assuming "HWC" layout).)code",
        std::vector<int>{0, 1})
    .AddOptionalArg("dim_names",
        R"code(If provided, it specifies the indexes of the dimensions that `crop` argument represents as described in
the layout.
Example: `crop=(200,400)` together with `dim_names="WH"` represents `crop_h=200` and `crop_w=400` for any layout.
Example: `crop=(10,400,200)` together with `dim_names="DHW"` represents `crop_d=10`, `crop_h=400` and `crop_w=200` for any layout.
If provided, `dim_names` takes higher priority than `dims`)code",
        "HW");

}  // namespace dali
