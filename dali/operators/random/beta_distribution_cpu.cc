// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/operators/random/beta_distribution.h"
#include "dali/operators/random/rng_base_cpu.h"

namespace dali {

DALI_SCHEMA(random__Beta)
    .DocStr(R"code(Generates a random number from ``[0, 1]`` range following the beta distribution.

The beta distribution has the following probabilty distribution function:

.. math:: f(x) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}

where ``Г`` is the gamma function defined as:

.. math:: \Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} \, dx

The operator supports ``float32`` and ``float64`` output types.

The shape of the generated data can be either specified explicitly with a `shape` argument,
or chosen to match the shape of the `__shape_like` input, if provided. If none are present,
a single value per sample is generated.
)code")
    .NumInput(0, 1)
    .InputDox(0, "shape_like", "TensorList",
              "Shape of this input will be used to infer the shape of the output, if provided.")
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AddOptionalArg("alpha", R"code(The alpha parameter, a positive ``float32`` scalar.)code", 1.0f,
                    true)
    .AddOptionalArg("beta", R"code(The beta parameter, a positive ``float32`` scalar.)code", 1.0f,
                    true)
    .AddParent("RNGAttr");

DALI_REGISTER_OPERATOR(random__Beta, BetaDistribution<CPUBackend>, CPU);

}  // namespace dali
