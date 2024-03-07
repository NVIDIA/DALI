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

#include "dali/pipeline/data/types.h"
#include "dali/test/operators/name_dump.h"

namespace dali {

DALI_REGISTER_OPERATOR(NameDump, NameDump, CPU);

DALI_SCHEMA(NameDump)
    .DocStr("Operator for testing naming utilities and propagation of API information to backend.")
    .AddOptionalArg("target", "\"module\" or \"op_name\"", "op_name")
    .AddOptionalArg("include_module", "If module should be included in the \"op_name\"", false)
    .NumInput(0)
    .NumOutput(1);


DALI_REGISTER_OPERATOR(sub__NameDump, NameDump, CPU);

DALI_SCHEMA(sub__NameDump)
    .DocStr("Operator for testing naming utilities and propagation of API information to backend.")
    .NumInput(0)
    .NumOutput(1)
    .AddParent("NameDump");

DALI_REGISTER_OPERATOR(sub__sub__NameDump, NameDump, CPU);

DALI_SCHEMA(sub__sub__NameDump)
    .DocStr("Operator for testing naming utilities and propagation of API information to backend.")
    .NumInput(0)
    .NumOutput(1)
    .AddParent("NameDump");

}  // namespace dali
