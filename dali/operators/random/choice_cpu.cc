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

#include "dali/operators/random/choice.h"
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(random__Choice)
    .DocStr(R"code(Generates a random sample from a given 1D array.

The probability of selecting a sample from the input is determined by the corresponding probability
specified in `p` argument.

The shape of the generated data can be either specified explicitly with a `shape` argument,
or chosen to match the shape of the `__shape_like` input, if provided. If none are present,
a single value per sample is generated.

The type of the output matches the type of the input.
For scalar inputs, only integral types are supported, otherwise any type can be used.
The operator supports selection from an input containing elements of one of DALI enum types,
that is: :meth:`nvidia.dali.types.DALIDataType`, :meth:`nvidia.dali.types.DALIImageType`, or
:meth:`nvidia.dali.types.DALIInterpType`.
)code")
    .NumInput(1, 2)
    .InputDox(0, "a", "scalar or TensorList",
              "If a scalar value `__a` is provided, the operator behaves as if "
              "``[0, 1, ..., __a-1]`` list was passed as input. "
              "Otherwise `__a` is treated as 1D array of input samples.")
    .InputDox(1, "shape_like", "TensorList",
              "Shape of this input will be used to infer the shape of the output, if provided.")
    .InputDevice(1, InputDevice::Metadata)
    .NumOutput(1)
    .AddOptionalArg<std::vector<float>>("p",
                                        "Distribution of the probabilities. "
                                        "If not specified, uniform distribution is assumed.",
                                        nullptr, true)
    .AddOptionalArg<std::vector<int>>("shape", "Shape of the output data.", nullptr, true)
    .AddRandomSeedArg();

DALI_REGISTER_OPERATOR(random__Choice, Choice<CPUBackend>, CPU);


}  // namespace dali
