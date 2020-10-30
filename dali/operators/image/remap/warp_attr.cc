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

Non-integer sizes are rounded to nearest integer. The channel dimension should
be excluded (for example, for RGB images, specify ``(480,640)``, not ``(480,640,3)``.
)code",
      vector<float>(), true)
  .AddOptionalArg("fill_value", R"code(Value used to fill areas that are outside the source image.

If a value is not specified, the source coordinates are clamped and the border pixel is
repeated.)code", 0.0f)
  .AddOptionalArg("dtype",
      R"code(Output data type.

If not set, the input type is used.)code",
      DALI_NO_TYPE)
  .DeprecateArgInFavorOf("output_dtype", "dtype")  // deprecated since 0.24dev
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR);

}  // namespace dali
