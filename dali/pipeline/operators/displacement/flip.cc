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


#include "dali/pipeline/operators/displacement/flip.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Flip, Flip<CPUBackend>, CPU);

DALI_SCHEMA(Flip)
    .DocStr("Flip the image on the horizontal and/or vertical axes.")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("horizontal",
        R"code(Perform a horizontal flip. Default value is 1.)code", 1, true)
    .AddOptionalArg("vertical",
        R"code(Perform a vertical flip. Default value is 0.)code", 0, true)
    .AddParent("DisplacementFilter");

}  // namespace dali

