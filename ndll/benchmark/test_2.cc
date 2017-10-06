#include <chrono>

#include "ndll/pipeline/operators/hybrid_jpg_decoder.h"
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
  shared_ptr<HybridJPEGDecodeChannel> decode_channel(new HybridJPEGDecodeChannel);
  ndll::HuffmanDecoder<CPUBackend> huffman_decoder(decode_channel);

  huffman_decoder.set_num_threads(num_thread);
  huffman_decoder.set_batch_size(batch_size);

  Batch<CPUBackend> *batch = CreateJPEGBatch<CPUBackend>(
      jpegs_, jpeg_sizes_, batch_size);
  Batch<CPUBackend> output_batch;

  // Get the dims for the output batch
  huffman_decoder.SetOutputType(&output_batch, batch->type());

  int tid = 0;
  vector<Dims> output_shape(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    Datum<CPUBackend> datum(batch, i);
    output_shape[i] = huffman_decoder.InferOutputShape(datum, i, tid);
  }
  output_batch.Resize(output_shape);

  // Run the op
  high_resolution_clock::time_point t1, t2;
  t1 = high_resolution_clock::now();
  int iters = 100;
  for (int j = 0; j < iters; ++j) {
    for (int i = 0; i < batch_size; ++i) {
      Datum<CPUBackend> datum(batch, i);
      Datum<CPUBackend> out_datum(&output_batch, i);
      huffman_decoder.Run(datum, &out_datum, i, tid);
    }
  }
  t2 = high_resolution_clock::now();
  float frames_per_ns = iters*batch_size /
    float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
  cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}
