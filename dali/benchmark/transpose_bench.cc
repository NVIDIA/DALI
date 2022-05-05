// Copyright (c) 2022, Konrad Litwiński. All rights reserved.
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

#include <benchmark/benchmark.h>
#include "dali/benchmark/operator_bench.h"
#include "dali/benchmark/dali_bench.h"

namespace dali {

static std::vector<int> dimensions = {3, 80, 1000};

static std::vector<int> permutations3[] = {
  {0, 1, 2},
  {2, 0, 1},
  {2, 1, 0},
};

static void TransposeGPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
    for (int H : dimensions) {
      for (int W : dimensions) {
        for (int C : dimensions) {
          if (24000 <= H * W * C && H * W * C <= 1000000) {
            for (unsigned i = 0; i < sizeof(permutations3) / sizeof(*permutations3); i++) {
              b->Args({batch_size, H, W, C, i});
            }
          }
        }
      }
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, TransposeGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);
  auto perm = permutations3[st.range(4)];

  this->RunGPU<uint8_t>(
    st,
    OpSpec("Transpose")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("perm", perm),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, TransposeGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(TransposeGPUArgs);

}  // namespace dali
