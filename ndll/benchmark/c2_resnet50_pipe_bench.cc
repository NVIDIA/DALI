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
 
  // Create the pipeline
  Pipeline<PinnedCPUBackend, GPUBackend> pipe(
      batch_size,
      num_thread,
      main_stream,
      0);

  shared_ptr<Batch<PinnedCPUBackend>> batch(CreateJPEGBatch<PinnedCPUBackend>(
          this->jpegs_, this->jpeg_sizes_, batch_size));
  shared_ptr<Batch<GPUBackend>> output_batch(new Batch<GPUBackend>);
  
  // Add the data reader
  BatchDataReader<PinnedCPUBackend> reader(batch);
  pipe.AddDataReader(reader);
  
  // Add a decoder and some transformers
  bool color = true;
  TJPGDecoder<PinnedCPUBackend> jpg_decoder(color);
  pipe.AddDecoder(jpg_decoder);

  // Add a resize+crop+mirror op
  if (fast_resize) {
    FastResizeCropMirrorOp<PinnedCPUBackend> resize_crop_mirror_op(
        true, false, 256, 480, true, 224, 224, 0.5f);
    pipe.AddPrefetchOp(resize_crop_mirror_op);
  } else {
    ResizeCropMirrorOp<PinnedCPUBackend> resize_crop_mirror_op(
        true, false, 256, 480, true, 224, 224, 0.5f);
    pipe.AddPrefetchOp(resize_crop_mirror_op);
  }

  // Add normalize permute op
  NormalizePermuteOp<GPUBackend, float> norm_permute_op(
      {128, 128, 128}, {1, 1, 1}, 224, 224, 3);
  pipe.AddForwardOp(norm_permute_op);
  
  // Build and run the pipeline
  pipe.Build(output_batch);

  // Run once to allocate the memory
  // pipe.Print();
  pipe.RunPrefetch();
  pipe.RunCopy();
  pipe.RunForward();
  
  while(st.KeepRunning()) {
    pipe.RunPrefetch();
    pipe.RunCopy();
    pipe.RunForward();
    CUDA_CALL(cudaDeviceSynchronize());
  }
  
  // DEBUG
  // DumpCHWImageBatchToFile<float>(*output_batch);
  st.counters["FPS"] = benchmark::Counter(batch_size*st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK_DEFINE_F(NDLLBenchmark, C2HybridResNet50Pipeline)(benchmark::State& st) {
  int batch_size = st.range(0);
  int num_thread = st.range(1);
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
 
  // Create the pipeline
  Pipeline<PinnedCPUBackend, GPUBackend> pipe(
      batch_size,
      num_thread,
      main_stream,
      0);
  
  shared_ptr<Batch<PinnedCPUBackend>> batch(CreateJPEGBatch<PinnedCPUBackend>(
          this->jpegs_, this->jpeg_sizes_, batch_size));
  shared_ptr<Batch<GPUBackend>> output_batch(new Batch<GPUBackend>);
  
  // Add the data reader
  BatchDataReader<PinnedCPUBackend> reader(batch);
  pipe.AddDataReader(reader);
  
  // Add a hybrid jpeg decoder
  shared_ptr<HybridJPEGDecodeChannel> decode_channel(new HybridJPEGDecodeChannel);
  HuffmanDecoder<PinnedCPUBackend> huffman_decoder(decode_channel);
  pipe.AddDecoder(huffman_decoder);

  DCTQuantInvOp<GPUBackend> idct_op(true, decode_channel);
  pipe.AddForwardOp(idct_op);

  // Add a batched resize op
  ResizeOp<GPUBackend> resize_op(true, false, 256, 480, true, NDLL_INTERP_LINEAR);
  pipe.AddForwardOp(resize_op);

  // Add a bached crop+mirror+normalize+permute op
  CropMirrorNormalizePermuteOp<GPUBackend, float> final_op(
      true, 224, 224, 0.5f, true, {128, 128, 128}, {1, 1, 1});
  pipe.AddForwardOp(final_op);
  
  // Build and run the pipeline
  pipe.Build(output_batch);

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
    CUDA_CALL(cudaDeviceSynchronize());
  }
  NDLLProfilerStop();

  // DEBUG
  // DumpCHWImageBatchToFile<float16>(*output_batch);
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
  // for (int batch_size = 32; batch_size <= 32; batch_size += 32) {
  //   for (int num_thread = 1; num_thread <= 4; ++num_thread) {
  //     b->Args({batch_size, num_thread});
  //   }
  // }
  b->Args({32, 1});
}

BENCHMARK_REGISTER_F(NDLLBenchmark, C2HybridResNet50Pipeline)->Iterations(1)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Apply(HybridPipeArgs);

} // namespace ndll
