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

#include "dali/operators/generic/one_hot.h"

namespace dali {

void OneHot::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &input = ws.InputRef<CPUBackend>(0);
  // check
  for (int i = 0; i < batch_size_; ++i) {
    auto &inptr = input[i];
    auto cls = inptr.template mutable_data<int>();
    // DALI_ENFORCE(*cls < nclasses_, "Value ", cls,
    //     " is bigger than specified number of classes");
    auto &outptr = output[i];
    auto one_hot = outptr.template mutable_data<int>();
    one_hot[*cls] = 1;
  }
}

// Check
DALI_REGISTER_OPERATOR(OneHot, OneHot, CPU);

// Check
DALI_SCHEMA(OneHot)
    .DocStr(
        "Produce tensor representing one hot encoding "
        " of the given input")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("nclasses", R"code(Number of all classes)code", 0);

}  // namespace dali