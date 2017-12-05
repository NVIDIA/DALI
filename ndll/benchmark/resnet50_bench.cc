#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_bench.h"
#include "ndll/pipeline/pipeline.h"

namespace ndll {

class RN50Bench : public NDLLBenchmark {
public:
protected:
};

BENCHMARK_DEFINE_F(RN50Bench, C2Pipeline)(benchmark::State& st) {
  bool fast_resize = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  NDLLImageType img_type = NDLL_RGB;

  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      0);

  TensorList<CPUBackend> data;
  this->MakeJPEGBatch(&data, batch_size);
  pipe.AddExternalInput("raw_jpegs");
  pipe.SetExternalInput("raw_jpegs", data);
  
  pipe.AddOperator(
      OpSpec("TJPGDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", img_type)
      .AddInput("raw_jpegs", "cpu")
      .AddOutput("images", "cpu")
      );

  // Add a resize+crop+mirror op
  if (fast_resize) {
    pipe.AddOperator(
        OpSpec("FastResizeCropMirror")
        .AddArg("device", "cpu")
        .AddArg("random_resize", true)
        .AddArg("warp_resize", false)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", true)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5f)
        .AddInput("images", "cpu")
        .AddOutput("resized", "cpu")
        );
  } else {
    pipe.AddOperator(
        OpSpec("ResizeCropMirror")
        .AddArg("device", "cpu")
        .AddArg("random_resize", true)
        .AddArg("warp_resize", false)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", true)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5f)
        .AddInput("images", "cpu")
        .AddOutput("resized", "cpu")
        );
  }

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
      .AddOutput("final_batch", "gpu")
      );
  
  // Build and run the pipeline
  vector<std::pair<string, string>> outputs = {{"final_batch", "gpu"}};
  pipe.Build(outputs);

  // Run once to allocate the memory
  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  while(st.KeepRunning()) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  }
  
  st.counters["FPS"] = benchmark::Counter(batch_size*st.iterations(), benchmark::Counter::kIsRate);
}

static void PipeArgs(benchmark::internal::Benchmark *b) {
  for (int fast_resize = 0; fast_resize < 2; ++fast_resize) {
    for (int batch_size = 32; batch_size <= 32; batch_size += 32) {
      for (int num_thread = 1; num_thread <= 4; ++num_thread) {
        b->Args({fast_resize, batch_size, num_thread});
      }
    }
  }
}

BENCHMARK_REGISTER_F(RN50Bench, C2Pipeline)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

} // namespace ndll
