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

#include "dali/test/operators/passthrough.h"

#include <cstdlib>

namespace dali {

DALI_REGISTER_OPERATOR(PassthroughOp, PassthroughOp<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(PassthroughOp, PassthroughOp<GPUBackend>, GPU);

DALI_SCHEMA(PassthroughOp)
    .DocStr("Operator that always passes inputs to outputs as pass through, used for testing.")
    .NumInput(1)
    .NumOutput(1)
    .PassThrough({{0, 0}});

}  // namespace dali
