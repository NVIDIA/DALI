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

struct ColorTwistArgs {
    float bri;
    float con;
    float hue;
    float sat;
};

ColorTwistArgs kArgs{0.1, 1.1, 10., 0.1};

BENCHMARK_DEFINE_F(OperatorBench, ColorTwistGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(1);
  int C = 3;

  this->RunGPU<uint8_t>(
    st,
    OpSpec("ColorTwist")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("brightness", kArgs.bri)
      .AddArg("contrast", kArgs.con)
      .AddArg("hue", kArgs.hue)
      .AddArg("saturation", kArgs.sat),
    batch_size, H, W, C);
}

BENCHMARK_DEFINE_F(OperatorBench, OldColorTwistGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(1);
  int C = 3;

  this->RunGPU<uint8_t>(
    st,
    OpSpec("OldColorTwist")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("brightness", kArgs.bri)
      .AddArg("contrast", kArgs.con)
      .AddArg("hue", kArgs.hue)
      .AddArg("saturation", kArgs.sat),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, ColorTwistGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Ranges({{1, 128}, {128, 2048}});

BENCHMARK_REGISTER_F(OperatorBench, OldColorTwistGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Ranges({{1, 128}, {128, 2048}});

}  // namespace dali
