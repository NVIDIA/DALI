// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/random/choice.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(random__Choice)
    .DocStr(R"code(Generates a random sample from a given 1-D array.

The probability of generating a value 1 (true) is determined by the ``probability`` argument.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single value per
sample is generated.
)code")
    .NumInput(1, 2)
    .InputDox(0, "a", "scalar or tensor",
        R"code(If a scalar value is provided, values between [0, ``a``) are sampled. Otherwise
``a`` is treated as 1-D array of input samples.)code")
    .InputDox(1, "shape_like", "", "")
    .NumOutput(1)
    .AddOptionalArg<bool>("replace", "", true)
    .AddOptionalArg<int>("axis", "", true)
    .AddOptionalArg<float>("p",
      R"code(Distribution of the probabilities)code",
      nullptr, true)
    .AddParent("RNGAttr");

DALI_REGISTER_OPERATOR(random__Choice, Choice<CPUBackend>, CPU);


}  // namespace dali
