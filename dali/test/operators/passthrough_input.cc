// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/test/operators/passthrough_input.h"

namespace dali {

DALI_SCHEMA(PassthroughInput)
                .DocStr(
                        R"code(
The operator that is a passthrough operator and also an input operator. Used for test only.
)code")
                .NumInput(0)
                .NumOutput(1)
                .AddParent("InputOperatorBase");


DALI_REGISTER_OPERATOR(PassthroughInput, PassthroughInput<MixedBackend>, Mixed);

}  // namespace dali
