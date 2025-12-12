// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/test/operators/deprecation_warning.h"

namespace dali {

DALI_REGISTER_OPERATOR(DeprecationWarningOp, DeprecationWarningOp, CPU);

DALI_SCHEMA(DeprecationWarningOp)
    .DocStr("Operator for deprecation warnings.")
    .NumInput(0)
    .NumOutput(1)
    .Deprecate("1.0", "", "Additional message");


DALI_REGISTER_OPERATOR(sub__DeprecationWarningOp, DeprecationWarningOp, CPU);

DALI_SCHEMA(sub__DeprecationWarningOp)
    .DocStr("Operator for deprecation warnings.")
    .NumInput(0)
    .NumOutput(1)
    .Deprecate("1.0", "sub__sub__DeprecationWarningOp", "Another message");

DALI_REGISTER_OPERATOR(sub__sub__DeprecationWarningOp, DeprecationWarningOp, CPU);

DALI_SCHEMA(sub__sub__DeprecationWarningOp)
    .DocStr("Operator for deprecation warnings.")
    .NumInput(0)
    .NumOutput(1)
    .Deprecate("1.0", "sub__sub__DeprecationWarningOp");

}  // namespace dali
