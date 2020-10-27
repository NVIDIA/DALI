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

#include <benchmark/benchmark.h>
#include "dali/benchmark/operator_bench.h"
#include "dali/benchmark/dali_bench.h"

namespace dali {

BENCHMARK_DEFINE_F(OperatorBench, PreemphasisGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int N = st.range(1);

  this->RunGPU<uint8_t>(
    st,
    OpSpec("PreemphasisFilter")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu"),
    batch_size, TensorShape<1>{N, }, "t");
}

BENCHMARK_REGISTER_F(OperatorBench, PreemphasisGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Ranges({{1, 256}, {1000, 50000}});

}  // namespace dali
