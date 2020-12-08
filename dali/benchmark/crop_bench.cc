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

static void CropCPUArgs(benchmark::internal::Benchmark *b) {
  int batch_size = 1;
  for (int H = 1000; H >= 500; H /= 2) {
    int W = H, C = 3;
    int crop_h = static_cast<float>(9*H/10);
    int crop_w = static_cast<float>(9*W/10);
    b->Args({batch_size, H, W, C, crop_h, crop_w});
  }
}

BENCHMARK_DEFINE_F(OperatorBench, CropCPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);
  int crop_h = st.range(4);
  int crop_w = st.range(5);

  this->RunCPU<uint8_t>(
    st,
    OpSpec("Crop")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("crop", std::vector<float>{static_cast<float>(crop_h), static_cast<float>(crop_w)})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, CropCPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(CropCPUArgs);

static void CropGPUArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 128; batch_size >= 1; batch_size /= 4) {
    for (int H = 2000; H >= 500; H /= 2) {
      int W = H, C = 3;
      int crop_h = static_cast<float>(5*H/10);
      int crop_w = static_cast<float>(5*W/10);
      b->Args({batch_size, H, W, C, crop_h, crop_w});
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, CropGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);
  int crop_h = st.range(4);
  int crop_w = st.range(5);

  this->RunGPU<uint8_t>(
    st,
    OpSpec("Crop")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("crop", std::vector<float>{static_cast<float>(crop_h), static_cast<float>(crop_w)})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, CropGPU)->Iterations(1000)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(CropGPUArgs);

}  // namespace dali
