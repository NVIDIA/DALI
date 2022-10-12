// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/test/operators/copy.h"
#include "dali/pipeline/data/types.h"

namespace dali {

DALI_REGISTER_OPERATOR(CopyArgumentOp, CopyArgumentOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(CopyArgumentOp, CopyArgumentOp<GPUBackend>, GPU);

DALI_SCHEMA(CopyArgumentOp)
    .DocStr("Operator that always copies argument input to output, used for testing.")
    .NumInput(0)
    .AddArg("to_copy", "Input to be copied, expected to be argument input", DALI_FLOAT, true)
    .NumOutput(1);

}  // namespace dali
