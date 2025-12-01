// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/test/dali_test_config.h"

namespace dali {

class C2Alexnet : public DALIBenchmark {
};

BENCHMARK_DEFINE_F(C2Alexnet, Caffe2Pipe)(benchmark::State& st) { // NOLINT
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
      0, -1, pipelined, 2,
      async);

  dali::string path(testing::dali_extra_path() + "/db/c2lmdb");
  pipe.AddOperator(
      OpSpec("Caffe2Reader")
      .AddArg("device", "cpu")
      .AddArg("path", path)
      .AddOutput("compressed_images", StorageDevice::CPU)
      .AddOutput("labels", StorageDevice::CPU));

  pipe.AddOperator(
      OpSpec("ImageDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("compressed_images", StorageDevice::CPU)
      .AddOutput("images", StorageDevice::CPU));

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform1", StorageDevice::CPU));

  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform2", StorageDevice::CPU));

  // Add coin flip RNG for mirror mask
  pipe.AddOperator(
      OpSpec("CoinFlip")
      .AddArg("device", "cpu")
      .AddArg("probability", 0.5f)
      .AddOutput("mirror", StorageDevice::CPU));

  // Add a resize+crop+mirror op
  pipe.AddOperator(
      OpSpec("FastResizeCropMirror")
      .AddArg("device", "cpu")
      .AddArg("resize_x", 256)
      .AddArg("resize_y", 256)
      .AddArg("crop", vector<float>{224, 224})
      .AddArg("mirror_prob", 0.5f)
      .AddInput("images", StorageDevice::CPU)
      .AddArgumentInput("crop_pos_x", "uniform1")
      .AddArgumentInput("crop_pos_y", "uniform2")
      .AddArgumentInput("mirror", "mirror")
      .AddOutput("resized", StorageDevice::CPU));

  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT16)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddInput("resized", StorageDevice::GPU)
      .AddOutput("final_batch", StorageDevice::GPU));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final_batch", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.Run();
    }
    pipe.Run();
    pipe.Outputs(&ws);

    if (st.iterations() == st.max_iterations && pipelined) {
      // Block for the last batch to finish
      pipe.Outputs(&ws);
    }
  }

  WriteCHWBatch<float16>(ws.Output<GPUBackend>(0), 128, 1, "img");
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

BENCHMARK_REGISTER_F(C2Alexnet, Caffe2Pipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(C2Alexnet, HybridPipe)(benchmark::State& st) { // NOLINT
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
      OpSpec("CaffeReader")
      .AddArg("device", "cpu")
      .AddArg("path", "/data/imagenet-compressed/256px/ilsvrc12_train_lmdb")
      .AddOutput("compressed_images", StorageDevice::CPU)
      .AddOutput("labels", StorageDevice::CPU));

  // Add a hybrid jpeg decoder
  pipe.AddOperator(
      OpSpec("ImageDecoder")
      .AddArg("device", "mixed")
      .AddInput("compressed_images", StorageDevice::CPU)
      .AddArg("output_type", img_type)
      .AddOutput("images", StorageDevice::GPU));

  // Add a batched resize op
#if 0
  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddArg("random_resize", false)
      .AddArg("warp_resize", false)
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 256)
      .AddArg("image_type", img_type)
      .AddArg("interp_type", DALI_INTERP_LINEAR)
      .AddInput("images", StorageDevice::GPU)
      .AddOutput("resized", StorageDevice::GPU));
#endif

  // Add uniform RNG
  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform1", StorageDevice::CPU));

  pipe.AddOperator(
      OpSpec("Uniform")
      .AddArg("device", "cpu")
      .AddArg("range", vector<float>{0, 1})
      .AddOutput("uniform2", StorageDevice::CPU));

  // Add coin flip RNG for mirror mask
  pipe.AddOperator(
      OpSpec("CoinFlip")
      .AddArg("device", "cpu")
      .AddArg("probability", 0.5f)
      .AddOutput("mirror", StorageDevice::CPU));

  // Add a batched crop+mirror+normalize+permute op
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("dtype", DALI_FLOAT16)
      .AddArg("crop", vector<float>{224, 224})
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddInput("images", StorageDevice::GPU)
      .AddArgumentInput("crop_pos_x", "uniform1")
      .AddArgumentInput("crop_pos_y", "uniform2")
      .AddArgumentInput("mirror", "mirror")
      .AddOutput("final", StorageDevice::GPU));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  while (st.KeepRunning()) {
    if (st.iterations() == 1 && pipelined) {
      // We will start he processing for the next batch
      // immediately after issueing work to the gpu to
      // pipeline the cpu/copy/gpu work
      pipe.Run();
    }
    pipe.Run();
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

BENCHMARK_REGISTER_F(C2Alexnet, HybridPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HybridPipeArgs);

}  // namespace dali
