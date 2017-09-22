#include <chrono>

#include "ndll/pipeline/operators/hybrid_decoder.h"
#include "ndll/pipeline/operators/normalize_permute_op.h"
#include "ndll/pipeline/operators/resize_crop_mirror_op.h"
#include "ndll/pipeline/operators/tjpg_decoder.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/util/image.h"
using namespace std::chrono;
using namespace ndll;

const string image_folder = "/home/tgale/data/raw-train-c2-480";

int main() {
  // Load the jpegs
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;
  LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  
  int batch_size = 32;
  int num_thread = 1;
  int num_stream = 8;
  cudaStream_t main_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
  bool stream_non_blocking = true;
 
  // Create the pipeline
  Pipeline<CPUBackend, GPUBackend> pipe(
      batch_size,
      num_thread,
      main_stream,
      num_stream,
      stream_non_blocking,
      0);

  // Add a hybrid jpeg decoder
  shared_ptr<HybridJPEGDecodeChannel> decode_channel(new HybridJPEGDecodeChannel);
  ndll::HuffmanDecoder<CPUBackend> huffman_decoder(decode_channel);
  pipe.AddDecoder(huffman_decoder);
    
  Batch<CPUBackend> *batch = CreateJPEGBatch<CPUBackend>(
      jpegs_, jpeg_sizes_, batch_size);
  Batch<GPUBackend> output_batch;
    
  // Build and run the pipeline
  pipe.Build(batch->type());

  // Run once to allocate the memory
  pipe.RunPrefetch(batch);
  pipe.RunCopy();
  pipe.RunForward(&output_batch);

  high_resolution_clock::time_point t1, t2;
  t1 = high_resolution_clock::now();

  int iters = 100;
  for (int i = 0; i < iters; ++i) {
    pipe.RunPrefetch(batch);
  }
  
  t2 = high_resolution_clock::now();
  // cout << "Whole loop time: " <<
  //   duration_cast<std::chrono::nanoseconds>(t2-t1).count()
  //      << " ns" << endl;

  float frames_per_ns = iters*batch_size /
    float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
  cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}
