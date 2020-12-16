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

static void CropMirrorNormalizeCPUArgs(benchmark::internal::Benchmark *b) {
  int batch_size = 8;
  int mean = 128, std = 1;
  for (auto &dtype : {DALI_FLOAT}) {
    for (auto nchw : {0, 1}) {
      for (int mirror : {0, 1}) {
        for (int pad : {0, 1}) {
          for (int H = 1000; H >= 500; H /= 2) {
            int W = H, C = 3;
            int crop_h = static_cast<float>(9 * H / 10);
            int crop_w = static_cast<float>(9 * W / 10);
            b->Args({dtype, nchw, mirror, pad,
                     batch_size, H, W, C, crop_h, crop_w,
                     mean, std});
          }
        }
      }
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, CropMirrorNormalizeCPU)(benchmark::State& st) {
  DALIDataType dtype = static_cast<DALIDataType>(st.range(0));
  int nchw = static_cast<int>(st.range(1));
  int mirror = st.range(2);
  int pad = st.range(3);
  int batch_size = st.range(4);
  int H = st.range(5);
  int W = st.range(6);
  int C = st.range(7);
  int crop_h = st.range(8);
  int crop_w = st.range(9);
  float mean = static_cast<float>(st.range(10));
  float std = static_cast<float>(st.range(11));

  this->RunCPU<uint8_t>(
    st,
    OpSpec("CropMirrorNormalize")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "cpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("output_layout", nchw ? "CHW" : "HWC")
      .AddArg("dtype", dtype)
      .AddArg("crop", std::vector<float>{static_cast<float>(crop_h), static_cast<float>(crop_w)})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f)
      .AddArg("mean", std::vector<float>(C, mean))
      .AddArg("std", std::vector<float>(C, std))
      .AddArg("mirror", mirror),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, CropMirrorNormalizeCPU)->Iterations(500)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(CropMirrorNormalizeCPUArgs);

static void CropMirrorNormalizeGPUArgs(benchmark::internal::Benchmark *b) {
  int mean = 128, std = 1;
  for (int batch_size : {1, 8, 128, 256}) {
    for (int H = 1000; H >= 250; H /= 2) {
      int W = H, C = 3;
      int crop_h = static_cast<float>(9 * H / 10);
      int crop_w = static_cast<float>(9 * W / 10);
      for (auto &dtype : {DALI_FLOAT}) {
        for (auto nchw : {0, 1}) {
          for (int mirror : {0, 1}) {
            for (int pad : {0, 1}) {
              b->Args({batch_size, H, W, C, crop_h, crop_w,
                       dtype, nchw, mirror,
                       pad, mean, std});
            }
          }
        }
      }
    }
  }
}

BENCHMARK_DEFINE_F(OperatorBench, CropMirrorNormalizeGPU)(benchmark::State& st) {
  int batch_size = st.range(0);
  int H = st.range(1);
  int W = st.range(2);
  int C = st.range(3);
  int crop_h = st.range(4);
  int crop_w = st.range(5);
  DALIDataType dtype = static_cast<DALIDataType>(st.range(6));
  int nchw = static_cast<int>(st.range(7));
  int mirror = st.range(8);
  int pad = st.range(9);
  float mean = static_cast<float>(st.range(10));
  float std = static_cast<float>(st.range(11));

  this->RunGPU<uint8_t>(
    st,
    OpSpec("CropMirrorNormalize")
      .AddArg("max_batch_size", batch_size)
      .AddArg("num_threads", 1)
      .AddArg("device", "gpu")
      .AddArg("output_type", DALI_RGB)
      .AddArg("output_layout", nchw ? "CHW" : "HWC")
      .AddArg("dtype", dtype)
      .AddArg("crop", std::vector<float>{static_cast<float>(crop_h), static_cast<float>(crop_w)})
      .AddArg("crop_pos_x", 0.5f)
      .AddArg("crop_pos_y", 0.5f)
      .AddArg("mean", std::vector<float>(C, mean))
      .AddArg("std", std::vector<float>(C, std))
      .AddArg("mirror", mirror),
    batch_size, H, W, C);
}

BENCHMARK_REGISTER_F(OperatorBench, CropMirrorNormalizeGPU)->Iterations(500)
->Unit(benchmark::kMicrosecond)
->UseRealTime()
->Apply(CropMirrorNormalizeGPUArgs);

}  // namespace dali
