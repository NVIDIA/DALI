// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/random/noise/salt_and_pepper_noise.h"
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(noise__SaltAndPepper)
    .DocStr(R"code(Applies salt-and-pepper noise to the input.

The shape and data type of the output will match the input.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<float>("prob",
      R"code(Probability of an output value to take a salt or pepper value.)code",
      0.05f, true)
    .AddOptionalArg<float>("salt_vs_pepper",
      R"code(Probability of a corrupted output value to take a salt value.)code",
      0.5f, true)
    .AddOptionalArg<float>("salt_val",
      R"code(Value of "salt".

If not provided, the salt value will be 1.0 for floating point types or the
maximum value of the input data type otherwise, converted to the data type of the input.)code",
      nullptr, true)
    .AddOptionalArg<float>("pepper_val",
      R"code(Value of "pepper".

If not provided, the pepper value will be -1.0 for floating point types or the
minimum value of the input data type otherwise, converted to the data type of the input.)code",
      nullptr, true)
    .AddOptionalArg<bool>("per_channel",
      R"code(Determines whether the noise should be generated for each channel independently.

If set to True, the noise is generated for each channel independently,
resulting in some channels being corrupted and others kept intact. If set to False, the noise
is generated once and applied to all channels, so that all channels in a pixel should either be
kept intact, take the "pepper" value, or the "salt" value.

Note: Per-channel noise generation requires the input layout to contain a channels ('C') dimension,
or be empty. In the case of the layout being empty, channel-last layout is assumed.)code",
      false);

DALI_REGISTER_OPERATOR(noise__SaltAndPepper, SaltAndPepperNoise<CPUBackend>, CPU);

}  // namespace dali
