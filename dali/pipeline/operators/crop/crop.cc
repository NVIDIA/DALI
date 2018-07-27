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

#include "dali/pipeline/operators/crop/crop.h"

namespace dali {

DALI_SCHEMA(Crop)
  .DocStr(R"code(Perform a random crop.)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("crop_pos_x",
      R"code(Horizontal position of the crop in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .AddOptionalArg("crop_pos_y",
      R"code(Vertical position of the crop in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image)code", DALI_RGB, false)
  .AddArg("crop",
      R"code(Size of the cropped image. If only a single value `c` is provided,
 the resulting crop will be square with size `(c,c)`)code", DALI_INT_VEC)
  .EnforceInputLayout(DALI_NHWC);

}  // namespace dali
