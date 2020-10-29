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

#include "dali/operators/generic/shapes.h"

namespace dali {

DALI_SCHEMA(Shapes)
    .DocStr(R"code(Returns the shapes of inputs.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowSequences()
    .SupportVolumetric()
    .AddOptionalArg("dtype", R"code(Data type to which the sizes are converted.)code", DALI_INT64)
    .DeprecateArgInFavorOf("type", "dtype");  // deprecated since 0.27dev

DALI_REGISTER_OPERATOR(Shapes, Shapes<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Shapes, Shapes<GPUBackend>, GPU);

}  // namespace dali
