// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_bench.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

class Alexnet : public NDLLBenchmark {
};

BENCHMARK_DEFINE_F(Alexnet, CaffePipe)(benchmark::State& st) { // NOLINT
  int executor = st.range(0);
  bool fast_resize = st.range(1);
  int batch_size = st.range(2);
  int num_thread = st.range(3);
  NDLLImageType img_type = NDLL_RGB;

  bool pipelined = executor > 0;
  bool async = executor > 1;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0, pipelined,
      async);

  ndll::string path(std::getenv("NDLL_TEST_CAFFE_LMDB_PATH"));
  pipe.AddOperator(
      OpSpec("CaffeReader")
      .AddArg("device", "cpu")
      .AddArg("path", path)
      .AddOutput("compressed_images", "cpu")
      .AddOutput("labels", "cpu"));

  pipe.AddOperator(
      OpSpec("TJPGDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("compressed_images", "cpu")
      .AddOutput("images", "cpu"));

  // Add a resize+crop+mirror op
  pipe.AddOperator(
      OpSpec("ResizeCropMirror")
      .AddArg("device", "cpu")
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 256)
      .AddArg("random_crop", true)
      .AddArg("crop_h", 224)
      .AddArg("crop_w", 224)
      .AddArg("mirror_prob", 0.5f)
      .AddInput("images", "cpu")
      .AddOutput("resized", "cpu"));

  pipe.AddOperator(
      OpSpec("NormalizePermute")
      .AddArg("device", "gpu")
      .AddArg("output_type", NDLL_FLOAT16)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddArg("height", 224)
      .AddArg("width", 224)
      .AddArg("channels", 3)
      .AddInput("resized", "gpu")
      .AddOutput("final_batch", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final_batch", "gpu"}};
  pipe.Build(outputs);

  string serialized = pipe.SerializeToProtobuf();

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

  WriteCHWBatch<float16>(*ws.Output<GPUBackend>(0), 128, 1, "img");
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

BENCHMARK_REGISTER_F(Alexnet, CaffePipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

BENCHMARK_DEFINE_F(Alexnet, HybridPipe)(benchmark::State& st) { // NOLINT
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
      OpSpec("CaffeReader")
      .AddArg("device", "cpu")
      .AddArg("path", "/data/imagenet-compressed/256px/ilsvrc12_train_lmdb")
      .AddOutput("compressed_images", "cpu")
      .AddOutput("labels", "cpu"));

  // Add a hybrid jpeg decoder
  pipe.AddOperator(
      OpSpec("HuffmanDecoder")
      .AddArg("device", "cpu")
      .AddInput("compressed_images", "cpu")
      .AddOutput("dct_data", "cpu")
      .AddOutput("jpeg_meta", "cpu"));

  pipe.AddOperator(
      OpSpec("DCTQuantInv")
      .AddArg("device", "gpu")
      .AddArg("output_type", img_type)
      .AddInput("dct_data", "gpu")
      .AddInput("jpeg_meta", "cpu")
      .AddOutput("images", "gpu"));

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
      .AddArg("interp_type", NDLL_INTERP_LINEAR)
      .AddInput("images", "gpu")
      .AddOutput("resized", "gpu"));
#endif

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
      .AddInput("images", "gpu")
      .AddOutput("final", "gpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final", "gpu"}};
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

  // WriteCHWBatch<float16>(*ws.Output<GPUBackend>(0), 128, 1, "img");
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

BENCHMARK_REGISTER_F(Alexnet, HybridPipe)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HybridPipeArgs);

}  // namespace ndll
