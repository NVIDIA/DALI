// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_bench.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll { // NOLINT

class MakeContiguousBench : public NDLLBenchmark { // NOLINT
};

BENCHMARK_DEFINE_F(MakeContiguousBench, CoalescedMemcpy)(benchmark::State& st) { // NOLINT
  int tensor_size = st.range(0);
  bool coalesced = (st.range(1) == 1);
  int num_thread = st.range(2);

  int batch_size = 256;
  bool pipelined = true;
  bool async = true;

  Pipeline pipe(
      batch_size,
      num_thread,
      0, -1, pipelined,
      async);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);

  pipe.AddOperator(
      OpSpec("HostDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", NDLL_RGB)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu"));

  pipe.AddOperator(
      OpSpec("FastResizeCropMirror")
      .AddArg("device", "cpu")
      .AddArg("random_resize", true)
      .AddArg("warp_resize", false)
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 480)
      .AddArg("random_crop", true)
      .AddArg("crop_h", tensor_size)
      .AddArg("crop_w", tensor_size)
      .AddArg("mirror_prob", 0.5f)
      .AddInput("images", "cpu")
      .AddOutput("resized", "cpu"));

  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"resized", "gpu"}};
  pipe.Build(outputs, coalesced);

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

  int num_batches = st.iterations() + static_cast<int>(pipelined);
  st.counters["FPS"] = benchmark::Counter(batch_size*num_batches,
      benchmark::Counter::kIsRate);
}

static void PipeArgs(benchmark::internal::Benchmark *b) {
    for (int tensor_size = 16; tensor_size <= 128; tensor_size += 16) {
      for (int coalesced = 0; coalesced <= 1; ++coalesced) {
        for (int num_thread = 1; num_thread <= 1; ++num_thread) {
          b->Args({(tensor_size > 224 ? 224 : tensor_size), coalesced, num_thread});
        }
      }
    }
}


BENCHMARK_REGISTER_F(MakeContiguousBench, CoalescedMemcpy)->Iterations(1)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

}  // namespace ndll
