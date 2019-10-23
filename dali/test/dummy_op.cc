// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dummy_op.h"

#include <cstdlib>

namespace dali {

DALI_REGISTER_OPERATOR(DummyOp, DummyOp<CPUBackend>, CPU);

DALI_SCHEMA(DummyOp)
  .DocStr("Dummy operator for testing")
  .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
  .NumInput(0, 10)
  .AddOptionalArg("num_outputs",
      R"code(Number of outputs.)code", 2);

}  // namespace dali
