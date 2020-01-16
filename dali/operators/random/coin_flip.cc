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


#include "dali/operators/random/coin_flip.h"

namespace dali {

void CoinFlip::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  for (int i = 0; i < batch_size_; ++i) {
    output[i].mutable_data<int>()[0] = dis_(rng_) ? 1 : 0;
  }
}

DALI_REGISTER_OPERATOR(CoinFlip, CoinFlip, CPU);

DALI_SCHEMA(CoinFlip)
  .DocStr("Produce tensor filled with 0s and 1s - results of random coin flip,"
      " usable as an argument for select ops.")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("probability",
      R"code(Probability of returning 1.)code", 0.5f);

}  // namespace dali
