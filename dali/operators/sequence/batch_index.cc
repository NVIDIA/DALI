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
#include "dali/operators/sequence/batch_index.h"

namespace dali {

DALI_SCHEMA(BatchIndex)
  .DocStr(R"(Produces a sequence of integers starting at 0 indexing batch.)")
  .NumInput(0)
  .NumOutput(1);

void BatchIndex::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  int N = GetBatchSize(ws);
  if (N < 1)
    return;
  auto out_view = view<int, 0>(output);

  for (int i = 0; i < N; ++i) {
    out_view.data[i][0] = i;
  }
}

DALI_REGISTER_OPERATOR(BatchIndex, BatchIndex, CPU);

}  // namespace dali
