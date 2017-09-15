#include <benchmark/benchmark.h>

#include "ndll/benchmark/ndll_main_bench.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"

namespace ndll {

BENCHMARK_DEFINE_F(NDLLBenchmark, C2ResNet50Pipeline)(benchmark::State& st) {
  int batch_size = 4;
  int num_thread = 1;
  int num_stream = 8;
  cudaStream_t main_stream = 0;
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

  // Add a dump image op
  DumpImageOp<CPUBackend> dump_image_op;
  pipe.AddPrefetchOp(dump_image_op);
    
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
  pipe.Print();
  pipe.RunPrefetch(batch);
  pipe.RunCopy();
  pipe.RunForward(&output_batch);

  DumpCHWImageBatchToFile<float>(output_batch);

  while(st.KeepRunning()) {
    
  }
}

BENCHMARK_REGISTER_F(NDLLBenchmark, C2ResNet50Pipeline)->Iterations(10);

} // namespace ndll
