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

#include "optical_flow.h"

namespace dali {

const std::string kPresetArgName = "preset";   // NOLINT
const std::string kOutputFormatArgName = "output_format";   // NOLINT
const std::string kEnableHintsArgName = "enable_hints";   // NOLINT

DALI_SCHEMA(OpticalFlow)
                .DocStr(R"code(Calculates the Optical Flow for sequence of images given as a input.
 Mandatory input for the operator is a sequence of frames.
 As as optional input, operator accepts external hints for OF calculation.
The output format of this operator matches output format of OF driver API)code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(kPresetArgName, R"code(Setting quality level of OF calculation.
 0.0f ... 1.0f, where 1.0f is best quality, lowest speed)code", .0f, false)
                .AddOptionalArg(kOutputFormatArgName,
                                R"code(Setting grid size for output vector.)code", -1, false)
                .AddOptionalArg(kEnableHintsArgName,
                                R"code(enabling/disabling temporal hints for sequences longer than 2 images)code",
                                false, false);

DALI_REGISTER_OPERATOR(OpticalFlow, OpticalFlow<CPUBackend>, CPU);

}  // namespace dali