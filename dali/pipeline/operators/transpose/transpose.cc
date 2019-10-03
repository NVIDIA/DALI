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

#include "dali/pipeline/operators/transpose/transpose.h"

namespace dali {

DALI_SCHEMA(Transpose)
  .DocStr("Transpose tensor dimension to a new permutated dimension specified by `perm`.")
  .NumInput(1)
  .NumOutput(1)
  .AllowSequences()
  .AddArg("perm",
      R"code(Permutation of the dimensions of the input (e.g. [2, 0, 1]).)code",
      DALI_INT_VEC)
  .AddOptionalArg("transpose_layout",
      R"code(When set to true, the output data layout will be transposed according to perm.
Otherwise, the input layout is copied to the output)code",
      true)
  .AddOptionalArg("output_layout",
      R"code(If provided, sets output data layout, overriding any `transpose_layout` setting)code",
      "");

}  // namespace dali
