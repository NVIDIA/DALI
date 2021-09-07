// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/operators/random/uniform_distribution.h"

namespace dali {

DALI_SCHEMA(random__Uniform)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

It can be configured to produce a continuous uniform distribution in the ``range`` [min, max),
or a discrete uniform distribution where any of the specified ``values`` [v0, v1, ..., vn] occur
with equal probability.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a scalar is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range ``[min, max)`` of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.

.. warning::
  When specifying an integer type as ``dtype``, the generated numbers can go outside
  the specified range, due to rounding.
)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values [v0, v1, ..., vn] produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr");

DALI_REGISTER_OPERATOR(random__Uniform, UniformDistribution<CPUBackend>, CPU);

// Deprecated alias
DALI_SCHEMA(Uniform)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

It can be configured to produce a continuous uniform distribution in the ``range`` [min, max),
or a discrete uniform distribution where any of the specified ``values`` [v0, v1, ..., vn] occur
with equal probability.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a scalar is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range ``[min, max)`` of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.

.. warning::
  When specifying an integer type as ``dtype``, the generated numbers can go outside
  the specified range, due to rounding.
)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values [v0, v1, ..., vn] produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr")
    .Deprecate("random__Uniform");  // Deprecated in 0.30


DALI_REGISTER_OPERATOR(Uniform, UniformDistribution<CPUBackend>, CPU);

}  // namespace dali
