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

#include <vector>
#include "dali/operators/random/uniform.h"

namespace dali {

void Uniform::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  for (int i = 0; i < batch_size_; ++i) {
    auto *sample_data = output[i].mutable_data<float>();
    auto sample_len = output[i].size();
    for (int k = 0; k < sample_len; ++k) {
      sample_data[k] = dis_(rng_);
    }
  }
}

DALI_REGISTER_OPERATOR(Uniform, Uniform, CPU);

DALI_SCHEMA(Uniform)
  .DocStr("Produce tensor filled with uniformly distributed random numbers.")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("range",
    R"code(Range of produced random numbers.)code", std::vector<float>({-1, 1}))
  .AddOptionalArg("shape",
    R"code(Shape of the samples)code", std::vector<int>{1});

}  // namespace dali
