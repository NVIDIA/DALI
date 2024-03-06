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

#include "dali/test/operators/origin_trace_dump.h"

#include <cstdlib>
#include <string>

namespace dali {

DALI_REGISTER_OPERATOR(OriginTraceDump, OriginTraceDump, CPU);

DALI_SCHEMA(OriginTraceDump)
    .DocStr("Operator for testing origin stack trace from Python.")
    .NumInput(0, 10)
    .NumOutput(1);

}  // namespace dali
