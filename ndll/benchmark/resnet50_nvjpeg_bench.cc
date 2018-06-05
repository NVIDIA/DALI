// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_bench.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

class RealRN50 : public NDLLBenchmark {
};

BENCHMARK_DEFINE_F(RealRN50, nvjpegPipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  NDLLImageType img_type = NDLL_RGB;

  bool pipelined = executor > 0;
  bool async = executor > 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, pipelined,
      async);

  pipe.AddOperator(
    OpSpec("Caffe2Reader")
    .AddArg("path", "/data/imagenet/train-c2lmdb-480")
    .AddOutput("raw_jpegs", "cpu")
    .AddOutput("labels", "cpu"));

  pipe.AddOperator(
    OpSpec("nvJPEGDecoder")
    .AddArg("device", "mixed")
    .AddArg("output_type", img_type)
    .AddArg("max_streams", num_thread)
    .AddArg("use_batched_decode", false)
    .AddInput("raw_jpegs", "cpu")
    .AddOutput("images", "gpu"));


  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddArg("random_resize", true)
      .AddArg("warp_resize", false)
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 480)
      .AddArg("image_type", img_type)
      .AddArg("interp_type", NDLL_INTERP_LINEAR)
      .AddInput("images", "gpu")
      .AddOutput("resized", "gpu"));

  // Add a bached crop+mirror+normalize+permute op
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("output_type", NDLL_FLOAT16)
      .AddArg("random_crop", true)
      .AddArg("crop", vector<int>{224, 224})
      .AddArg("mirror_prob", 0.5f)
      .AddArg("image_type", img_type)
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

  WriteHWCBatch<uint8_t>(*ws.Output<GPUBackend>(0), 0, 1, "img");
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


BENCHMARK_DEFINE_F(RealRN50, HybridPipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  NDLLImageType img_type = NDLL_RGB;

  bool pipelined = executor > 0;
  bool async = executor > 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, pipelined,
      async);

  pipe.AddOperator(
    OpSpec("Caffe2Reader")
    .AddArg("path", "/data/imagenet/train-c2lmdb-480")
    .AddOutput("raw_jpegs", "cpu")
    .AddOutput("labels", "cpu"));

  pipe.AddOperator(
      OpSpec("HuffmanDecoder")
      .AddArg("device", "cpu")
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("dct_data", "cpu")
      .AddOutput("jpeg_meta", "cpu"));

  pipe.AddOperator(
      OpSpec("DCTQuantInv")
      .AddArg("device", "gpu")
      .AddArg("output_type", img_type)
      .AddInput("dct_data", "gpu")
      .AddInput("jpeg_meta", "cpu")
      .AddOutput("images", "gpu"));

  pipe.AddOperator(
      OpSpec("Resize")
      .AddArg("device", "gpu")
      .AddArg("random_resize", true)
      .AddArg("warp_resize", false)
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 480)
      .AddArg("image_type", img_type)
      .AddArg("interp_type", NDLL_INTERP_LINEAR)
      .AddInput("images", "gpu")
      .AddOutput("resized", "gpu"));

  // Add a bached crop+mirror+normalize+permute op
  pipe.AddOperator(
      OpSpec("CropMirrorNormalize")
      .AddArg("device", "gpu")
      .AddArg("output_type", NDLL_FLOAT16)
      .AddArg("random_crop", true)
      .AddArg("crop", vector<int>{224, 224})
      .AddArg("mirror_prob", 0.5f)
      .AddArg("image_type", img_type)
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

  WriteHWCBatch<uint8_t>(*ws.Output<GPUBackend>(0), 0, 1, "img");
  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(RealRN50, HybridPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

}  // namespace ndll
