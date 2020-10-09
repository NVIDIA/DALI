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

#include "dali/operators/generic/reduce/reduce.h"


namespace dali {

DALI_SCHEMA(ReduceBase)
  .AddOptionalArg(
    "axes",
    R"code(Axis or axes along which reduction is performed.

Not providing any axis results in reduction of all elements.)code",
    std::vector<int>{ })
  .AddOptionalArg(
    "keep_dims",
    "If True, maintains original input dimensins.",
    false);

DALI_REGISTER_OPERATOR(Sum, SumCPU, CPU);
DALI_SCHEMA(Sum)
  .DocStr("Sums input elements along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");


DALI_REGISTER_OPERATOR(Min, MinCPU, CPU);
DALI_SCHEMA(Min)
  .DocStr("Gets minimal input element along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");


DALI_REGISTER_OPERATOR(Max, MaxCPU, CPU);
DALI_SCHEMA(Max)
  .DocStr("Gets minimal input element along provided axes.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");

}  // namespace dali
