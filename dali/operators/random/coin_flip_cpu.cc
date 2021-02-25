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

#include "dali/operators/random/rng_base_cpu.h"
#include "dali/operators/random/coin_flip.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(random__CoinFlip)
    .DocStr(R"code(Generates random boolean values following a bernoulli distribution.

The probability of generating a value 1 (true) is determined by the ``probability`` argument.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single value per
sample is generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg<float>("probability",
      R"code(Probability of value 1.)code",
      0.5f, true)
    .AddParent("RNGAttr");

DALI_REGISTER_OPERATOR(random__CoinFlip, CoinFlip<CPUBackend>, CPU);

// Deprecated alias
DALI_SCHEMA(CoinFlip)
    .DocStr(R"code(Generates random boolean values following a bernoulli distribution.

The probability of generating a value 1 (true) is determined by the ``probability`` argument.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single value per
sample is generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddParent("random__CoinFlip")
    .Deprecate("random__CoinFlip");  // Deprecated in 0.30


DALI_REGISTER_OPERATOR(CoinFlip, CoinFlip<CPUBackend>, CPU);

}  // namespace dali
