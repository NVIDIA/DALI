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

void Uniform::AssignRange(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto curr_batch_size = ws.GetRequestedBatchSize(0);
  auto dist = std::uniform_real_distribution<float>(range_[0], range_[1]);
  for (int i = 0; i < curr_batch_size; ++i) {
    auto *sample_data = output[i].mutable_data<float>();
    auto sample_len = output[i].size();
    for (int k = 0; k < sample_len; ++k) {
      do {
        sample_data[k] = dist(rng_);
      } while (sample_data[k] >= range_[1]);  // Due to GCC and LLVM bug
    }
  }
}


void Uniform::AssignSet(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto curr_batch_size = ws.GetRequestedBatchSize(0);
  auto dist = std::uniform_int_distribution<int>(0, set_.size() - 1);
  for (int i = 0; i < curr_batch_size; ++i) {
    auto *sample_data = output[i].mutable_data<float>();
    auto sample_len = output[i].size();
    for (int k = 0; k < sample_len; ++k) {
      sample_data[k] = set_[dist(rng_)];
    }
  }
}


void Uniform::RunImpl(HostWorkspace &ws) {
  if (discrete_mode_) {
    AssignSet(ws);
  } else {
    AssignRange(ws);
  }
}

DALI_REGISTER_OPERATOR(Uniform, Uniform, CPU);

DALI_SCHEMA(Uniform)
  .DocStr(R"code(Produces random numbers following an uniform distribution.

Both continuous and discrete uniform distributions can be defined by providing
a continuous ``range`` or a discrete list of ``values``, respectively.

)code")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("range",
    R"code(Range (``[a, b)``) of produced random numbers.

This argument is mutually exclusive with ``values``.
)code", std::vector<float>({-1, 1}))
  .AddOptionalArg("values",
    R"code(The discrete list of values from which the random numbers are picked.

This argument is mutually exclusive with ``range``.
)code", std::vector<float>({}))
  .AddOptionalArg("shape",
    R"code(Shape of the samples.)code", std::vector<int>{}, true);


}  // namespace dali
