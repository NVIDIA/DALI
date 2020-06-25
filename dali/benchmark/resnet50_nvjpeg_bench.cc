// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

namespace dali {

class RealRN50 : public DALIBenchmark {
};

BENCHMARK_DEFINE_F(RealRN50, nvjpegPipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  DALIImageType img_type = DALI_RGB;

  bool pipelined = executor > 0;
  bool async = executor > 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1, pipelined, 2,
      async);

  pipe.AddOperator(
    OpSpec("Caffe2Reader")
    .AddArg("path", "/data/imagenet/train-c2lmdb-480")
    .AddOutput("raw_jpegs", "cpu")
    .AddOutput("labels", "cpu"));

  pipe.AddOperator(
    OpSpec("ImageDecoder")
    .AddArg("device", "mixed")
    .AddArg("output_type", img_type)
    .AddArg("max_streams", num_thread)
    .AddArg("use_batched_decode", false)
    .AddInput("raw_jpegs", "cpu")
    .AddOutput("images", "gpu"));

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{256, 480})
      .AddOutput("resize", "cpu"));

  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddArg("image_type", img_type)
      .AddArg("interp_type", DALI_INTERP_LINEAR)
      .AddInput("images", "gpu")
      .AddArgumentInput("resize_shorter", "resize")
      .AddOutput("resized", "gpu"));

  // Add a bached crop+mirror+normalize+permute op
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT16)
      .AddArg("random_crop", true)
      .AddArg("crop", vector<float>{224, 224})
      .AddArg("mirror_prob", 0.5f)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddInput("resized", "gpu")
      .AddOutput("final", "gpu"));

// Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations && pipelined) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

#if DALI_DEBUG
  WriteHWCBatch(ws.Output<GPUBackend>(0), "img");
#endif
  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void PipeArgs(benchmark::internal::Benchmark *b) {
  for (int executor = 2; executor < 3; ++executor) {
    for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
      for (int num_thread = 1; num_thread <= 4; ++num_thread) {
        b->Args({executor, batch_size, num_thread});
      }
    }
  }
}

BENCHMARK_REGISTER_F(RealRN50, nvjpegPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);


}  // namespace dali
