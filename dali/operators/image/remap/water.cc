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


#include "dali/core/common.h"
#include "dali/operators/image/remap/water.h"
#include "dali/operators/image/remap/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Water, Water<CPUBackend>, CPU);

DALI_SCHEMA(Water)
    .DocStr("Performs a water augmentation, which makes the image appear to be underwater.")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("ampl_x",
        R"code(Amplitude of the wave in the x direction.)code", 10.f)
    .AddOptionalArg("ampl_y",
        R"code(Amplitude of the wave in the y direction.)code", 10.f)
    .AddOptionalArg<float>("freq_x",
        R"code(Frequency of the wave in the x direction.)code", 2.0 * M_PI / 128)
    .AddOptionalArg<float>("freq_y",
        R"code(Frequence of the wave in the y direction.)code", 2.0 * M_PI / 128)
    .AddOptionalArg("phase_x",
        R"code(Phase of the wave in the x direction.)code", 0.f)
    .AddOptionalArg("phase_y",
        R"code(Phase of the wave in the y direction.)code", 0.f)
    .InputLayout(0, "HWC")
    .AddParent("DisplacementFilter");

}  // namespace dali
