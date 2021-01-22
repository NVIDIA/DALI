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
#include "dali/core/random.h"

namespace dali {

DALI_SCHEMA(BatchPermutation)
  .DocStr(R"(Produces a batch of random integers which can be used as indices for
indexing samples in the batch.)")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("allow_repetitions",
      R"(If true, the output can contain repetitions and omissions.)", false)
  .AddOptionalArg("no_fixed_points", R"(If true, the the output permutation cannot contain fixed
points, that is ``out[i] != i``. This argument is ignored when batch size is 1.)", false);

void BatchPermutation::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  int N = ws.GetRequestedBatchSize(0);
  if (N < 1)
    return;
  auto out_view = view<int, 0>(output);

  bool rep = spec_.GetArgument<bool>("allow_repetitions");
  bool no_fixed = spec_.GetArgument<bool>("no_fixed_points") && N > 1;

  tmp_out_.resize(N);
  if (rep) {
    if (no_fixed)
      random_sequence_no_fixed_points(tmp_out_, 0, N, rng_);
    else
      random_sequence(tmp_out_, 0, N, rng_);
  } else {
    if (no_fixed)
      random_derangement(tmp_out_, rng_);
    else
      random_permutation(tmp_out_, rng_);
  }
  for (int i = 0; i < N; ++i) {
    out_view.data[i][0] = tmp_out_[i];
  }
}

DALI_REGISTER_OPERATOR(BatchPermutation, BatchPermutation, CPU);

}  // namespace dali
