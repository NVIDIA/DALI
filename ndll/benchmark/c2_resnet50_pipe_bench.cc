#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_main_bench.h"
#include "ndll/common.h"
#include "ndll/pipeline/operators/crop_mirror_normalize_permute_op.h"
#include "ndll/pipeline/operators/hybrid_jpg_decoder.h"
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/pipeline/operators/resize_op.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

BENCHMARK_DEFINE_F(NDLLBenchmark, C2ResNet50Pipeline)(benchmark::State& st) {
  bool fast_resize = st.range(0);
  int batch_size = st.range(1);
  int num_thread = st.range(2);
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
  NDLLImageType img_type = NDLL_RGB;
  
  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      main_stream,
      0);

  // Add the data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );

  pipe.AddDecoder(
      OpSpec("TJPGDecoder", "Prefetch")
      .AddArg("output_type", img_type)
      );

  // Add a resize+crop+mirror op
  if (fast_resize) {
    pipe.AddTransform(
        OpSpec("FastResizeCropMirrorOp", "Prefetch")
        .AddArg("random_resize", true)
        .AddArg("warp_resize", false)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", true)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5f)
        );
  } else {
    pipe.AddTransform(
        OpSpec("ResizeCropMirrorOp", "Prefetch")
        .AddArg("random_resize", true)
        .AddArg("warp_resize", false)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", true)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5f)
        );
  }

  pipe.AddTransform(
      OpSpec("NormalizePermuteOp", "Forward")
      .AddArg("output_type", NDLL_FLOAT16)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      .AddArg("height", 224)
      .AddArg("width", 224)
      .AddArg("channels", 3)
      );
  
  // Build and run the pipeline
  pipe.Build();

  // Run once to allocate the memory
  // pipe.Print();
  pipe.RunPrefetch();
  pipe.RunCopy();
  pipe.RunForward();
  
  while(st.KeepRunning()) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();
    CUDA_CALL(cudaStreamSynchronize(pipe.stream()));
  }
  
  // DEBUG
  // DumpCHWImageBatchToFile<float16>(pipe.output_batch());
  st.counters["FPS"] = benchmark::Counter(batch_size*st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK_DEFINE_F(NDLLBenchmark, C2HybridResNet50Pipeline)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
  NDLLImageType img_type = NDLL_RGB;
   
  // Create the pipeline
  Pipeline pipe(
      batch_size,
      num_thread,
      main_stream,
      0);
  
  // Add the data reader
  pipe.AddDataReader(
      OpSpec("BatchDataReader")
      .AddArg("jpeg_folder", image_folder)
      );
  
  // Add a hybrid jpeg decoder
  pipe.AddDecoder(
      OpSpec("HuffmanDecoder", "Prefetch")
      .AddExtraOutput("jpeg_meta")
      );
  
  pipe.AddTransform(
      OpSpec("DCTQuantInvOp", "Forward")
      .AddExtraInput("jpeg_meta")
      .AddArg("output_type", img_type)
      );
  
  // Add a batched resize op
  pipe.AddTransform(
      OpSpec("ResizeOp", "Forward")
      .AddArg("random_resize", true)
      .AddArg("warp_resize", false)
      .AddArg("resize_a", 256)
      .AddArg("resize_b", 480)
      .AddArg("image_type", img_type)
      .AddArg("interp_type", NDLL_INTERP_LINEAR)
      );

  // Add a bached crop+mirror+normalize+permute op
  pipe.AddTransform(
      OpSpec("CropMirrorNormalizePermuteOp", "Forward")
      .AddArg("output_type", NDLL_FLOAT16)
      .AddArg("random_crop", true)
      .AddArg("crop_h", 224)
      .AddArg("crop_w", 224)
      .AddArg("mirror_prob", 0.5f)
      .AddArg("image_type", img_type)
      .AddArg("mean", vector<float>{128, 128, 128})
      .AddArg("std", vector<float>{1, 1, 1})
      );
  
  // Build and run the pipeline
  pipe.Build();

  // Run once to allocate the memory
  // pipe.Print();
  pipe.RunPrefetch();
  pipe.RunCopy();
  pipe.RunForward();

  NDLLProfilerStart();
  while(st.KeepRunning()) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();
    CUDA_CALL(cudaStreamSynchronize(pipe.stream()));
  }
  NDLLProfilerStop();

  // DEBUG
  // DumpCHWImageBatchToFile<float16>(pipe.output_batch());
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

BENCHMARK_REGISTER_F(NDLLBenchmark, C2ResNet50Pipeline)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(PipeArgs);

static void HybridPipeArgs(benchmark::internal::Benchmark *b) {
  for (int batch_size = 32; batch_size <= 32; batch_size += 32) {
    for (int num_thread = 1; num_thread <= 4; ++num_thread) {
      b->Args({batch_size, num_thread});
    }
  }
}

BENCHMARK_REGISTER_F(NDLLBenchmark, C2HybridResNet50Pipeline)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HybridPipeArgs);

} // namespace ndll
