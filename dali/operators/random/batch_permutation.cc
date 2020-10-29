// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <random>
#include <vector>
#include "dali/operators/random/batch_permutation.h"

namespace dali {

DALI_SCHEMA(BatchPermutation)
  .DocStr(R"(Produces a batch of random integers which can be used as indices for
indexing samples in the batch.)")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("allow_repetitions",
      R"(If true, the output can contain repetitions and omissions.)", false);

void BatchPermutation::NoRepetitions(int N) {
  tmp_out_.resize(N);
  std::iota(tmp_out_.begin(), tmp_out_.end(), 0);
  std::shuffle(tmp_out_.begin(), tmp_out_.end(), rng_);
}

void BatchPermutation::WithRepetitions(int N) {
  std::uniform_int_distribution<int> dis(0, N-1);
  tmp_out_.resize(N);
  for (auto &x : tmp_out_)
    x = dis(rng_);
}

void BatchPermutation::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  int N = GetBatchSize(ws);
  if (N < 1)
    return;
  auto out_view = view<int, 0>(output);

  if (spec_.GetArgument<bool>("allow_repetitions"))
    WithRepetitions(N);
  else
    NoRepetitions(N);
  for (int i = 0; i < N; ++i) {
    out_view.data[i][0] = tmp_out_[i];
  }
}

DALI_REGISTER_OPERATOR(BatchPermutation, BatchPermutation, CPU);

}  // namespace dali

