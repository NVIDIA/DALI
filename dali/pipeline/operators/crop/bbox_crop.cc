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

namespace dali
{

DALI_SCHEMA(BBoxCrop)
    .DocStr(
        R"code(Perform a prospective crop to an image while keeping bounding boxes consistent.
        Crop is provided as two Tensors: `Begin` which contains the starting coordinates for the `crop` in `(x,y)` format,
        and 'Size' which contains the dimensions of the `crop` in `(w,h)` format. Bounding boxes are provided as a `(m*4)` Tensor,
        where each bounding box is represented as `[x,y,w,h]`.)code")
    .NumInput(2)
    .NumOutput(3)
    .AllowMultipleInputSets()
    .AddOptionalArg(
        "thresholds",
        R"code(Minimum overlap (IoU) with new crop to keep bounding boxes from being discarded.
    Selected at random for every sample from provided values.)code",
        std::vector<float>{0.})
    .AddOptionalArg(
        "aspect_ratio",
        R"code(Range `[min, max]` of valid aspect ratio values for new crops. Value for `min` should be greater or equal to `0.0`.)code",
        std::vector<float>{1., 1.})
    .AddOptionalArg(
        "scaling",
        R"code(Range `[min, max]` for crop size with respect to original image dimensions. Value for `min` should be greater or equal to `0.0`.)code",
        std::vector<float>{1., 1.})
    .EnforceInputLayout(DALI_NHWC);

DALI_REGISTER_OPERATOR(BBoxCrop, BBoxCrop, CPU);

} // namespace dali
