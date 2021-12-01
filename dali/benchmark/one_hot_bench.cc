// Copyright (c) 2021, Maksymilian Grochowski. All rights reserved.
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

static void OneHotGPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 256; batch_size >= 1; batch_size /= 2) {
    for (int H = 1000; H >= 500; H /= 2) {
      int W = H;
      int AXIS = 1;
      int NUM_CLASSES = 2;      
      b->Args({batch_size, H, W, AXIS, NUM_CLASSES});
      AXIS = 2;
      b->Args({batch_size, H, W, AXIS, NUM_CLASSES});
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, OneHotGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int axis = st.range(3);
  int num_classes = st.range(4);

  this->RunGPU<uint8_t>(st,
                        OpSpec("OneHot")
                          .AddArg("max_batch_size", batch_size)
                          .AddArg("num_threads", 1)
                          .AddArg("device", "gpu")
                          .AddArg("output_type", DALI_RGB)
                          .AddArg("axis", axis)
                          .AddArg("num_classes", num_classes),
                        batch_size, H, W);
}

BENCHMARK_REGISTER_F(OperatorBench, OneHotGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(OneHotGPUArgs);

}  // namespace dali 