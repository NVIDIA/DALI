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

#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(WarpAttr)
  .DocStr(R"code(Apply an affine transformation to the image.)code")
  .AddOptionalArg<float>("size",
      R"code(Output size, in pixels/points.
Non-integer sizes are rounded to nearest integer.
Channel dimension should be excluded (e.g. for RGB images specify (480,640), not (480,640,3).)code",
      vector<float>(), true)
  .AddOptionalArg("fill_value", R"(Value used to fill areas that are outside source image.
If not specified, source coordinates are clamped and the border pixel is repeated.)",
      0.0f)
  .AddOptionalArg("output_dtype",
      R"code(Output data type. By default, same as input type)code",
      DALI_NO_TYPE)
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR);

}  // namespace dali
