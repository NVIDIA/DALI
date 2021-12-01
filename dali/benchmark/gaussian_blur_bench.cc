// Copyright (c) 2021, Justyna Micota. All rights reserved.
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

static void GaussianBlurGPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
    for (int H = 1000; H >= 500; H /= 2) {
      int W = H, C = 3;
      b->Args({batch_size, H, W, C});
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, GaussianBlurGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);
  // no change in performace for different value of sigma:
  float sigma = 1.0;

  this->RunGPU<uint8_t>(
    st,
    OpSpec("GaussianBlur")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("sigma", sigma),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, GaussianBlurGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(GaussianBlurGPUArgs);

}  // namespace dali
