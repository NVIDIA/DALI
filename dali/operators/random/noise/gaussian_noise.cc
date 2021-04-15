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

#include "dali/operators/random/noise/gaussian_noise.h"
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(noise__Gaussian)
    .DocStr(R"code(Applies gaussian noise to the input.

The shape and data type of the output will match the input.
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<float>("mean",
      R"code(Mean of the distribution.)code",
      0.f, true)
    .AddOptionalArg<float>("stddev",
      R"code(Standard deviation of the distribution.)code",
      1.f, true);

DALI_REGISTER_OPERATOR(noise__Gaussian, GaussianNoise<CPUBackend>, CPU);

}  // namespace dali
