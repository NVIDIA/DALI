// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_bench.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

class FileReaderAlexnet : public NDLLBenchmark {
};

BENCHMARK_DEFINE_F(FileReaderAlexnet, CaffePipe)(benchmark::State& st) { // NOLINT
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
      0, -1, pipelined,
      async);

  ndll::string list_root(std::getenv("NDLL_TEST_FILE_READER_LIST_ROOT"));
  pipe.AddOperator(
      OpSpec("FileReader")
      .AddArg("device", "cpu")
      .AddArg("file_root", list_root)
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
      .AddArg("crop", vector<int>{224, 224})
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

BENCHMARK_REGISTER_F(FileReaderAlexnet, CaffePipe)->Iterations(1)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

}  // namespace ndll
