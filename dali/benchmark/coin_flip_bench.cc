// Copyright (c) 2021, Szymon Karpiński. All rights reserved.
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

BENCHMARK_DEFINE_F(OperatorBench, CoinFlipGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(1);

  this->RunGPU<uint8_t>(st,
                        OpSpec("CoinFlip")
                          .AddArg("max_batch_size", batch_size)
                          .AddArg("num_threads", 1)
                          .AddArg("device", "gpu")
                          .AddInput("data", StorageDevice::GPU),
                        batch_size, H, W, 1);
}

BENCHMARK_REGISTER_F(OperatorBench, CoinFlipGPU)->Iterations(100)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->ArgsProduct({
  benchmark::CreateRange(1, 256, 2),
  {500, 1000, 2000},
});

}  // namespace dali
