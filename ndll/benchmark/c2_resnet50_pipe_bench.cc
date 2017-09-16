#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_main_bench.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

// TODO(tgale): Add arguments and metrics for framerate
BENCHMARK_DEFINE_F(NDLLBenchmark, C2ResNet50Pipeline)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  int num_stream = st.range(2);
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
  bool stream_non_blocking = true;
 
  // Create the pipeline
  Pipeline<CPUBackend, GPUBackend> pipe(
      batch_size,
      num_thread,
      main_stream,
      num_stream,
      stream_non_blocking);
    
  // Add a decoder and some transformers
  bool color = true;
  TJPGDecoder<CPUBackend> jpg_decoder(color);
  pipe.AddDecoder(jpg_decoder);
    
  // Add a resize+crop+mirror op
  ResizeCropMirrorOp<CPUBackend> resize_crop_mirror_op(
      true, false, 256, 480, true, 224, 224, 0.5f);
  pipe.AddPrefetchOp(resize_crop_mirror_op);

  // Add normalize permute op
  NormalizePermuteOp<GPUBackend, float> norm_permute_op(
      {128, 128, 128}, {1, 1, 1}, 224, 224, 3);
  pipe.AddForwardOp(norm_permute_op);
    
  Batch<CPUBackend> *batch = CreateJPEGBatch<CPUBackend>(
      this->jpegs_, this->jpeg_sizes_, batch_size);
  Batch<GPUBackend> output_batch;
    
  // Build and run the pipeline
  pipe.Build(batch->type());

  // Run once to allocate the memory
  pipe.RunPrefetch(batch);
  pipe.RunCopy();
  pipe.RunForward(&output_batch);

  while(st.KeepRunning()) {
    pipe.RunPrefetch(batch);
    pipe.RunCopy();
    pipe.RunForward(&output_batch);
    CUDA_CALL(cudaDeviceSynchronize());
  }
  st.counters["FPS"] = benchmark::Counter(batch_size*st.iterations(), benchmark::Counter::kIsRate);
}

// static void ArgsToRun(benchmark::internal::Benchmark *b) {
//   for (int batch_size = 32; batch_size <= 256; batch_size += 32) {
//     for (int num_thread = 1; num_thread <= 8; ++num_thread) {
//       for (int num_stream = 4; num_stream <= 16; num_stream += 4) {
//         b->Args({batch_size, num_thread, num_stream});
//       }
//     }
//   }
// }

static void ArgsToRun(benchmark::internal::Benchmark *b) {
  for (int batch_size = 32; batch_size <= 32; batch_size += 32) {
    for (int num_thread = 1; num_thread <= 4; ++num_thread) {
        b->Args({batch_size, num_thread, 8});
    }
  }
}

BENCHMARK_REGISTER_F(NDLLBenchmark, C2ResNet50Pipeline)->Iterations(100)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(ArgsToRun);

} // namespace ndll
