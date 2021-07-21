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

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(RNGAttr)
    .DocStr(R"code(Random Number Generator attributes.

It should be added as parent to all RNG operators.)code")
    .AddOptionalArg<std::vector<int>>("shape",
      R"code(Shape of the output data.)code", nullptr, true)
    .AddOptionalArg<DALIDataType>("dtype",
      R"code(Output data type.

.. note::
  The generated numbers are converted to the output data type, rounding and clamping if necessary.
)code", nullptr);

}  // namespace dali
