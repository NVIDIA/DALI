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


#include "dali/pipeline/operators/displacement/warpaffine.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(WarpAffine, WarpAffine<CPUBackend>, CPU);

DALI_SCHEMA(WarpAffine)
    .DocStr(R"code(Apply an affine transformation to the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("matrix",
        R"code(Matrix of the transform (dst -> src).
Given list of values `(M11, M12, M13, M21, M22, M23)`
this operation will produce a new image using  formula

..

dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)

It is equivalent to OpenCV's `warpAffine` operation
with a flag `WARP_INVERSE_MAP` set.)code",
        DALI_FLOAT_VEC)
    .AddOptionalArg("use_image_center",
        R"code(Whether to use image center as the center of transformation.
When this is `True` coordinates are calculated from the center of the image.)code",
        false)
    .AddParent("DisplacementFilter");

}  // namespace dali
