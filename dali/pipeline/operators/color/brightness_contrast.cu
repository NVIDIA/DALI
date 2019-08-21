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

#include "brightness_contrast.h"

namespace dali {
namespace brightness_contrast {

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrast<CPUBackend>, CPU)

DALI_REGISTER_OPERATOR(BrightnessContrast, BrightnessContrast<GPUBackend>, GPU)


DALI_SCHEMA(BrightnessContrast)
                .DocStr(R"code( * HWC
)code")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg(detail::kOutputType, R"code(...)code", DALI_INT16)
                .AddOptionalArg(detail::kBrightness,
                                R"code(Set additive brightness delta. 0 denotes no-op)code", .0f,
                                true)
                .AddOptionalArg(detail::kContrast,
                                R"code(Set multiplicative contrast delta. 1 denotes no-op)code",
                                1.f, true);


}  // namespace brightness_contrast
}  // namespace dali
