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

DALI_SCHEMA(experimental__Remap)
    .DocStr(R"doc(
The remap operation applies a generic geometrical transformation to an image. In other words,
it takes pixel from one place in the input image and puts them in another place in
the output image. The transformation is described by ``mapx`` and ``mapy`` parameters, where:

    output(x,y) = input(mapx(x,y),mapy(x,y))

The type of the output tensor will match the type of the input tensor.

Handles only HWC layout.

Each ROI descriptor consists of 2 points: `lo` and `hi`. It is assumed, that `lo` denotes
upper-left corner of the ROI and `hi` denotes bottom-right corner. In both `lo` and `hi`,
first coordinate denotes value along x axis (i.e. width of the image) and the second
coordinate denotes value along y axis (i.e. height of the image). The (0, 0) point is the
upper-left corner of the image.
)doc")
    .NumInput(3)
    .NumOutput(1)
    .InputDox(0, "input", "TensorList", "Input data. Must be a 1- or 3-channel HWC image.")
    .InputDox(1, "mapx", "TensorList of float", "Defines the remap transformation for x coordinates.")
    .InputDox(2, "mapy", "TensorList of float", "Defines the remap transformation for y coordinates.")
    .AddOptionalArg("pixel_origin",R"doc(
Pixel origin. Possible values: ``"corner"``, ``"center"``.

Defines which part of the pixel (upper-left corner or center) is interpreted as its origin.
This value impacts the interpolation result. To match OpenCV, please pick ``"center"``.)doc","corner")
    .AddOptionalArg("interp","Interpolation type.",DALIInterpType::DALI_INTERP_LINEAR)
    .AddOptionalArg("roi","The region of interest",0)
//    .AddOptionalArg("border_type","Defines, how to act when a pixel would be drawn from outside of ROI.",boundary::BoundaryType::CONSTANT)
    .AddOptionalArg("border_value","Border policy",0)
    .AllowSequences();



}