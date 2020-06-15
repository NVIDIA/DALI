// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

DALI_SCHEMA(OutOfBoundsAttr)
    .DocStr(R"code(Out-of-bounds slicing attributes placeholder)code")
    .AddOptionalArg("out_of_bounds_policy",
        R"code(Determines the policy when slicing out of bounds of the input.
Supported values are:

- "error" (default) : Attempting to slice outside of the bounds of the image will produce an error.
- "pad": The input will be padded as needed with zeros or any other value specified with ``fill_values`` argument.
- "trim_to_shape": The slice window will be cut to the bounds of the input.))code", "error")
    .AddOptionalArg("fill_values",
        R"code(Determines padding values, only relevant if ``out_of_bounds_policy`` is set to "pad".
If a scalar is provided, it will be used for all the channels. If multiple values are given, there should be as many values as
channels (extent of dimension 'C' in the layout) in the output slice.)code", std::vector<float>{0.f});

}  // namespace dali
