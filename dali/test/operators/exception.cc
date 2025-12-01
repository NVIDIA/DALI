// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/test/operators/exception.h"
#include "dali/pipeline/data/types.h"

namespace dali {

DALI_REGISTER_OPERATOR(ThrowException, ThrowExceptionOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(ThrowException, ThrowExceptionOp<GPUBackend>, GPU);

DALI_SCHEMA(ThrowException)
    .DocStr("Operator that raises an exception.")
    .NumInput(0)
    .AddOptionalArg("exception_type", "Exception to be thrown", "TypeError")
    .AddOptionalArg("message", "Message to be reported in exception", "Test message")
    .AddOptionalArg<bool>("constructor", "Throw in constructor", false)
    .NumOutput(1);

}  // namespace dali
