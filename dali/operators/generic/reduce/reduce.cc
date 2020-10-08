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
    "",
    std::vector<int>{ })
  .AddOptionalArg(
    "keep_dims",
    "",
    false);

DALI_REGISTER_OPERATOR(Sum, Sum, CPU);
DALI_SCHEMA(Sum)
  .DocStr("Sum reduction operator.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");


DALI_REGISTER_OPERATOR(Min, Min, CPU);
DALI_SCHEMA(Min)
  .DocStr("Min reduction operator.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");


DALI_REGISTER_OPERATOR(Max, Max, CPU);
DALI_SCHEMA(Max)
  .DocStr("Max reduction operator.")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ReduceBase");

}  // namespace dali