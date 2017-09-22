#include <chrono>

#include "ndll/pipeline/operators/huffman_decoder.h"
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

  // Parse the jpegs
  vector<ParsedJpeg> parsed_jpegs(batch_size);

  int total = 0;
  for (int i = 0; i < batch_size; ++i) {
    JpegParserState state;
    cout << i << ": " << jpeg_sizes_[i] << endl;
    parseRawJpegHost(jpegs_[i], jpeg_sizes_[i], &state, &parsed_jpegs[i]);
    total += jpeg_sizes_[i];
  }
  cout << "total size: " << total << endl;
  
  // size the output buffer
  size_t out_size = 0;
  vector<int> offsets(batch_size*3);
  for (int i = 0; i < batch_size; ++i) {
    ParsedJpeg &jpeg = parsed_jpegs[i];
    for (int j = 0; j < jpeg.components; ++j) {
      int comp_id = i*3+j;
      offsets[comp_id] = out_size;
      out_size += jpeg.dctSize[i] / sizeof(int16);
      cout << out_size << endl;
    }
  }
  
  HuffmanDecoderState state;
  cout << "out_size: " << out_size << endl;
  vector<int16> output(out_size);

  // Run the op
  high_resolution_clock::time_point t1, t2;
  t1 = high_resolution_clock::now();
  int iters = 1000;
  for (int j = 0; j < iters; ++j) {
    for (int i = 0; i < batch_size; ++i) {
      ParsedJpeg &jpeg = parsed_jpegs[i];
      // Gather the pointers to each image
      vector<int16*> dct_coeff_ptrs(jpeg.components);

      for (int k = 0; k < jpeg.components; ++k) {
        dct_coeff_ptrs[k] = output.data() + offsets[i*3+k];
      }

      huffmanDecodeHost(jpeg, &state, &dct_coeff_ptrs);
    }
  }
  t2 = high_resolution_clock::now();
  float frames_per_ns = iters*batch_size /
    float(duration_cast<std::chrono::nanoseconds>(t2-t1).count());
  cout << "FPS: " << 1000000000 * frames_per_ns << endl;
}
