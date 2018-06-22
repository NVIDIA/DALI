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

#include "dali/pipeline/operators/support/random/uniform.h"

namespace dali {

void Uniform::RunImpl(SupportWorkspace * ws, const int idx) {
  DALI_ENFORCE(idx == 0, "Uniform does not support multiple input sets.");
  auto *output = ws->Output<CPUBackend>(idx);
  output->Resize({batch_size_});

  float * out_data = output->template mutable_data<float>();

  for (int i = 0; i < batch_size_; ++i) {
    out_data[i] = dis_(rng_);
  }
}

DALI_REGISTER_OPERATOR(Uniform, Uniform, Support);

DALI_SCHEMA(Uniform)
  .DocStr("Produce tensor filled with uniformly distributed random numbers.")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("range",
      R"code(Range of produced random numbers.)code", std::vector<float>({-1, 1}));

}  // namespace dali
