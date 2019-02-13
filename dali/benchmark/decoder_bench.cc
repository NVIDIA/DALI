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

class DecoderBench : public DALIBenchmark {
};

BENCHMARK_DEFINE_F(DecoderBench, HostDecoder)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("HostDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "cpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + 1;
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void HostPipeArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
    for (int num_thread = 1; num_thread <= 4; ++num_thread) {
      b->Args({batch_size, num_thread});
    }
  }
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoder)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HostPipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoder)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("use_batched_decode", false)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations();
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void nvJPEGPipeArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 128; batch_size <= 128; batch_size += 32) {
    for (int num_thread = 1; num_thread <= 4; ++num_thread) {
      b->Args({batch_size, num_thread});
    }
  }
}

BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoder)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(nvJPEGPipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, nvJPEGDecoderBatched)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", img_type)
      .AddArg("max_streams", num_thread)
      .AddArg("use_batched_decode", true)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations();
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}


BENCHMARK_REGISTER_F(DecoderBench, nvJPEGDecoderBatched)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(nvJPEGPipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderRandomCrop)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("HostDecoderRandomCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "cpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + 1;
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderRandomCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HostPipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderCrop)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("HostDecoderCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddArg("crop", std::vector<float>{224.0f, 224.0f})
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "cpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + 1;
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderCrop)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HostPipeArgs);

BENCHMARK_DEFINE_F(DecoderBench, HostDecoderSlice)(benchmark::State& st) { // NOLINT
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  DALIImageType img_type = DALI_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1,
      true,   // pipelined
      2,      // pipe length
      true);  // async

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  vector<Dims> shape(batch_size, {2});

  TensorList<CPUBackend> begin_data;
  begin_data.set_type(TypeInfo::Create<float>());
  begin_data.Resize(shape);
  float crop_x = 0.25f, crop_y = 0.124f;
  for (int k = 0; k < batch_size; k++) {
    begin_data.mutable_tensor<float>(k)[0] = crop_x;
    begin_data.mutable_tensor<float>(k)[1] = crop_y;
  }

  pipe.AddExternalInput("begin_data");
  pipe.SetExternalInput("begin_data", begin_data);

  TensorList<CPUBackend> crop_data;
  float crop_w = 0.5f, crop_h = 0.25f;
  crop_data.set_type(TypeInfo::Create<float>());
  crop_data.Resize(shape);
  for (int k = 0; k < batch_size; k++) {
    crop_data.mutable_tensor<float>(k)[0] = crop_w;
    crop_data.mutable_tensor<float>(k)[1] = crop_h;
  }

  pipe.AddExternalInput("crop_data");
  pipe.SetExternalInput("crop_data", crop_data);

  pipe.AddOperator(
      OpSpec("HostDecoderSlice")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddInput("begin_data", "cpu")
      .AddInput("crop_data", "cpu")
      .AddOutput("images", "cpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"images", "cpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.RunCPU();
      pipe.RunGPU();
    }
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  // WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
  int num_batches = st.iterations() + 1;
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(DecoderBench, HostDecoderSlice)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HostPipeArgs);

}  // namespace dali
