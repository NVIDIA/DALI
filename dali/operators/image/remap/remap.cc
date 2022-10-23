// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/remap/remap.h"

namespace dali::remap {

DALI_SCHEMA(Remap)  // TODO experimental
        .DocStr(R"doc(
The remap operation applies a generic geometrical transformation to an image. In other words,
it takes pixel from one place in the input image and puts them in another place in
the output image. The transformation is described by ``mapx`` and ``mapy`` parameters, where:

    output(x,y) = input(mapx(x,y),mapy(x,y))

The type of the output tensor will match the type of the input tensor.

Handles only HWC layout.

Currently picking border policy is not supported.
The ``DALIBorderType`` will always be ``CONSTANT`` with the value ``0``.
)doc")
        .NumInput(3)
        .NumOutput(1)
        .InputDox(0, "input", "TensorList", "Input data. Must be a 1- or 3-channel HWC image.")
        .InputDox(1, "mapx", "TensorList of float",
                  "Defines the remap transformation for x coordinates.")
        .InputDox(2, "mapy", "TensorList of float",
                  "Defines the remap transformation for y coordinates.")
        .AddOptionalArg<std::string>("pixel_origin", R"doc(
Pixel origin. Possible values: ``"corner"``, ``"center"``.

Defines which part of the pixel (upper-left corner or center) is interpreted as its origin.
This value impacts the interpolation result. To match OpenCV, please pick ``"center"``.)doc",
                                     "corner")
        .AddOptionalArg("interp", "Interpolation type.", DALI_INTERP_LINEAR)
        .AddOptionalArg<vector<float>>("roi_start", R"code(Origin of the input region of interest (ROI).

Must be specified together with ``roi_end``. The coordinates follow the tensor shape order, which is
the same as ``size``. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code", nullptr, true)
        .AddOptionalArg<vector<float>>("roi_end", R"code(End of the input region of interest (ROI).

Must be specified together with ``roi_start``. The coordinates follow the tensor shape order, which is
the same as ``size``. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code", nullptr, true)
        .AddOptionalArg("roi_relative", R"code(If true, ROI coordinates are relative to the input size,
where 0 denotes top/left and 1 denotes bottom/right)code", false)
//        .AddOptionalArg("roi", R"doc(
//The region of interest. Each ROI descriptor consists of 4 values: ``[lo_x, lo_y, hi_x, hi_y]``.
//It is assumed, that ``lo`` denotes upper-left corner of the ROI and `hi` denotes bottom-right corner.
//In both ``lo`` and ``hi``, first coordinate denotes value along x axis (i.e. width of the image)
//and the second coordinate denotes value along y axis (i.e. height of the image).
//The ``(0, 0)`` point is the upper-left pixel of the image.)doc",std::vector<int>{0,0,0,0}, true, true)
        .AllowSequences();


}