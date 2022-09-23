// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/benchmark/dali_bench.h"
#include "dali/test/dali_test_config.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"


namespace dali {

static void video_benchmark(benchmark::State& state) {
  // Init CUDA
  DeviceGuard(0);
  CUDA_CALL(cudaDeviceSynchronize());

  for (auto _ : state) {
    FramesDecoderGpu decoder(testing::dali_extra_path() + "/db/video/vfr/test_1.mp4");
    while (decoder.ReadNextFrame(nullptr, false)) {}
  }
}

BENCHMARK(video_benchmark);


}  // namespace dali
