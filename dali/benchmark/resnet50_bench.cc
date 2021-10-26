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
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

namespace dali {

class RN50 : public DALIBenchmark {
};

BENCHMARK_DEFINE_F(RN50, C2Pipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  bool fast_resize = st.range(1);
  int batch_size = st.range(2);
  int num_thread = st.range(3);
  DALIImageType img_type = DALI_RGB;

  bool pipelined = executor > 0;
  bool async = executor > 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1, pipelined, 3,
      async);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");

  pipe.AddOperator(
      OpSpec("ImageDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform1", "cpu"));

  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform2", "cpu"));

  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{256, 480})
      .AddOutput("resize", "cpu"));

  // Add coin flip RNG for mirror mask
  pipe.AddOperator(
      OpSpec("CoinFlip")
      .AddArg("device", "cpu")
      .AddArg("probability", 0.5f)
      .AddOutput("mirror", "cpu"));

  std::string resize_op = fast_resize ? "FastResizeCropMirror" : "ResizeCropMirror";
  // Add a resize+crop+mirror op
  pipe.AddOperator(
      OpSpec(resize_op)
      .AddArg("device", "cpu")
      .AddArg("crop", vector<float>{224, 224})
      .AddInput("images", "cpu")
      .AddArgumentInput("mirror", "mirror")
      .AddArgumentInput("crop_pos_x", "uniform1")
      .AddArgumentInput("crop_pos_y", "uniform2")
      .AddArgumentInput("resize_shorter", "resize")
      .AddOutput("resized", "cpu"));

  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT16)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddInput("resized", "gpu")
      .AddOutput("final_batch", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final_batch", "gpu"}};
  pipe.Build(outputs);

  pipe.SetExternalInput("raw_jpegs", data);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    pipe.SetExternalInput("raw_jpegs", data);
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.SetExternalInput("raw_jpegs", data);
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

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void PipeArgs(benchmark::internal::Benchmark *b) {
  for (int executor = 2; executor < 3; ++executor) {
    for (int fast_resize = 0; fast_resize < 2; ++fast_resize) {
      for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
        for (int num_thread = 1; num_thread <= 4; ++num_thread) {
          b->Args({executor, fast_resize, batch_size, num_thread});
        }
      }
    }
  }
}

BENCHMARK_REGISTER_F(RN50, C2Pipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(RN50, HybridPipe)(benchmark::State& st) { // NOLINT
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
      0, -1, pipelined, 3,
      async);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  // Add a hybrid jpeg decoder
  pipe.AddOperator(
      OpSpec("ImageDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{256, 480})
      .AddOutput("resize", "cpu"));

  // Add a batched resize op
  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddArg("image_type", img_type)
      .AddArg("interp_type", DALI_INTERP_LINEAR)
      .AddInput("images", "gpu")
      .AddArgumentInput("resize_shorter", "resize")
      .AddOutput("resized", "gpu"));

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform1", "cpu"));

  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform2", "cpu"));

  // Add coin flip RNG for mirror mask
  pipe.AddOperator(
      OpSpec("CoinFlip")
      .AddArg("device", "cpu")
      .AddArg("probability", 0.5f)
      .AddOutput("mirror", "cpu"));

  // Add a bached crop+mirror+normalize+permute op
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT16)
      .AddArg("random_crop", true)
      .AddArg("crop", vector<float>{224, 224})
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddInput("resized", "gpu")
      .AddArgumentInput("mirror", "mirror")
      .AddArgumentInput("crop_pos_x", "uniform1")
      .AddArgumentInput("crop_pos_y", "uniform2")
      .AddOutput("final", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final", "gpu"}};
  pipe.Build(outputs);

  pipe.SetExternalInput("raw_jpegs", data);


  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    pipe.SetExternalInput("raw_jpegs", data);
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.SetExternalInput("raw_jpegs", data);
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

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void HybridPipeArgs(benchmark::internal::Benchmark *b) {
  for (int executor = 0; executor < 3; ++executor) {
    for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
      for (int num_thread = 1; num_thread <= 4; ++num_thread) {
        b->Args({executor, batch_size, num_thread});
      }
    }
  }
}

BENCHMARK_REGISTER_F(RN50, HybridPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HybridPipeArgs);


BENCHMARK_DEFINE_F(RN50, nvJPEGPipe)(benchmark::State& st) { // NOLINT
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
      0, -1, pipelined, 3,
      async);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");

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

  // Add a batched resize op
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
  vector<std::pair<string, string>> outputs = {{"final", "gpu"}};
  pipe.Build(outputs);

  pipe.SetExternalInput("raw_jpegs", data);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    pipe.SetExternalInput("raw_jpegs", data);
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.SetExternalInput("raw_jpegs", data);
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

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void nvjpegPipeArgs(benchmark::internal::Benchmark *b) {
  for (int executor = 0; executor < 3; ++executor) {
    for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
      for (int num_thread = 1; num_thread <= 4; ++num_thread) {
        b->Args({executor, batch_size, num_thread});
      }
    }
  }
}

BENCHMARK_REGISTER_F(RN50, nvJPEGPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(nvjpegPipeArgs);


}  // namespace dali
