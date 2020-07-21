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
BENCHMARK_DEFINE_F(OperatorBench, NormalDistributionGPU)(benchmark::State& st) {
  const int batch_size = st.range(0);
  const int sample_dim = st.range(1);
  const bool single_value = st.range(2);

  auto spec = OpSpec("NormalDistribution")
      .AddArg("batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT)
      .AddArg("mean", 0.1f)
      .AddArg("stddev", 1.5f);

  if (!single_value) spec.AddInput("data", "gpu");

  this->RunGPU<float>(st, spec, batch_size, sample_dim, sample_dim, 3);
}

BENCHMARK_REGISTER_F(OperatorBench, NormalDistributionGPU)->Iterations(400)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Ranges({{16, 256}, {128, 512}, {0, 1}});

}  // namespace dali
