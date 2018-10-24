// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/crop/bbox_crop.h"

namespace dali {

DALI_SCHEMA(RandomBBoxCrop)
    .DocStr(
        R"code(Perform a prospective crop to an image while keeping bounding boxes consistent. Inputs must be supplied as two Tensors:
        `Images` containing image data in NHWC format, and `BBoxes` containing bounding boxes represented as `[l,t,r,b]`.
        Resulting prospective crop is provided as two Tensors: `Begin` containing the starting coordinates for the `crop` in `(x,y)` format,
        and 'Size' containing the dimensions of the `crop` in `(w,h)` format. Bounding boxes are provided as a `(m*4)` Tensor,
        where each bounding box is represented as `[l,t,r,b]` or `[x,y,w,h]`.)code")
    .NumInput(2)
    .NumOutput(3)
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum overlap (Intersection over union) of the bounding boxes with respect to the prospective crop.
    Selected at random for every sample from provided values. Default value is `[0.0]`, leaving the input image as-is in the new crop.)code",
        std::vector<float>{0.f})
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Range `[min, max]` of valid aspect ratio values for new crops. Value for `min` should be greater or equal to `0.0`.
        Default values are `[1.0, 1.0]`, disallowing changes in aspect ratio.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "scaling",
        R"code(Range `[min, max]` for crop size with respect to original image dimensions. Value for `min` should be greater or equal to `0.0`
        Default values are `[1.0, 1.0]`.)code",
        std::vector<float>{1.f, 1.f})
    .AddOptionalArg(
        "ltrb",
        R"code(If true, bboxes are returned as [left, top, right, bottom], else [x, y, width, height]. By default is set to `true`.)code",
        true)
    .AddOptionalArg(
        "num_attempts",
        R"code(Number of attempts to retrieve a patch with the desired parameters.)code",
        1)
    .EnforceInputLayout(DALI_NHWC);

DALI_REGISTER_OPERATOR(RandomBBoxCrop, RandomBBoxCrop, CPU);

}  // namespace dali
