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

#include "dali/pipeline/operators/color/hsv.h"

namespace dali {

//DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrast<CPUBackend>, CPU)
//DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrast<GPUBackend>, GPU)


DALI_SCHEMA(Hsv)
                .DocStr(R"code()code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(spec::kHue,
                                R"code(Set additive brightness delta. 0 denotes no-op)code", .0f,
                                true)
                .AddOptionalArg(spec::kSaturation,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true)
                .AddOptionalArg(spec::kValue,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true)
                .AddOptionalArg(spec::kOutputType, R"code(Set output data type)code", DALI_INT16);


}  // namespace dali
