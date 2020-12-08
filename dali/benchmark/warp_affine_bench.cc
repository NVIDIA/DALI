// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

static void WarpAffineCPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 16; batch_size >= 1; batch_size /= 2) {
    for (int H = 2048; H >= 256; H /= 2) {
      int W = H, C = 3;
      b->Args({batch_size, H, W, C});
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, WarpAffineCPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);

  vector<float> mtx = {
    1.1f, 0.2f, -10.0f,
    -0.15f, 0.9f, 5.0f
  };

  this->RunCPU<uint8_t>(
    st,
    OpSpec("WarpAffine")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 4)
      .AddArg("device", "cpu")
      .AddArg("matrix", mtx)
      .AddArg("fill_value", 42),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, WarpAffineCPU)->Iterations(50)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(WarpAffineCPUArgs);


static void WarpAffineGPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
    for (int H = 2048; H >= 256; H /= 2) {
      int W = H, C = 3;
      b->Args({batch_size, H, W, C});
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, WarpAffineGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);

  vector<float> mtx = {
    1.1f, 0.2f, -10.0f,
    -0.15f, 0.9f, 5.0f
  };

  this->RunGPU<uint8_t>(
    st,
    OpSpec("WarpAffine")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("matrix", mtx)
      .AddArg("fill_value", 42),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, WarpAffineGPU)->Iterations(100)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(WarpAffineGPUArgs);

}  // namespace dali
