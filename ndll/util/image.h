#ifndef NDLL_UTIL_IMAGE_H_
#define NDLL_UTIL_IMAGE_H_

#include <fstream>

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/batch.h"

// This file contains useful image utilities for reading and writing images.
// These functions are for testing and debugging, they should not be used
// inside the library as they throw exceptions on error.

namespace ndll {

/**
 * Loads jpegs from a specified image folder. Assumes the folder contains
 * a file 'image_list.txt' that lists all the different images in the 
 * folder
 */
void LoadJPEGS(string image_folder, vector<string> *jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes);

/**
 * Writes an HWC image to the specified file. Add the file extension '.txt'
 */
template <typename T>
void DumpHWCToFile(T *img, int h, int w, int c, int stride, string file_name) {
  CUDA_CALL(cudaDeviceSynchronize());
  T *tmp = new T[h*w*c];

  CUDA_CALL(cudaMemcpy2D(tmp, w*c*sizeof(T), img, stride*sizeof(T),
          w*c*sizeof(T), h, cudaMemcpyDefault));
  std::ofstream file(file_name + ".txt");
  NDLL_ENFORCE(file.is_open());

  file << h << " " << w << " " << c << endl;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << float(tmp[i*w*c + j*c + k]) << " ";
      }
    }
    file << endl;
  }
  delete[] tmp;
}

/**
 * Writes an CHW image to the specified file. Add the file extension '.txt'.
 * The data will be written in HWC format
 */
template <typename T>
void DumpCHWToFile(T *img, int h, int w, int c, string file_name) {
  CUDA_CALL(cudaDeviceSynchronize());
  T *tmp = new T[h*w*c];
    
  CUDA_CALL(cudaMemcpy(tmp, img, h*w*c*sizeof(T), cudaMemcpyDefault));
  std::ofstream file(file_name + ".txt");
  NDLL_ENFORCE(file.is_open());

  // write the image as HWC for our scripts
  file << h << " " << w << " " << c << endl;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << float(tmp[k*h*w + i*w +j]) << " ";
      }
    }
    file << endl;
  }
  delete[] tmp;
}

/**
 * Loads an image dumped by one of the previous two images
 */
void LoadFromFile(string file_name, uint8 **image, int *h, int *w, int *c);

/**
 * Creates a batch of jpegs from the input vector of jpegs. If not enough
 * jpegs are input, insert duplicates
 */
template <typename Backend>
auto CreateJPEGBatch(const vector<uint8*> &jpegs, const vector<int> &jpeg_sizes,
    int batch_size) -> Batch<Backend>* {
  NDLL_ENFORCE(jpegs.size() > 0);
  NDLL_ENFORCE(jpegs.size() == jpeg_sizes.size());
  Batch<Backend> *batch = new Batch<Backend>();

  // Create the shape
  vector<Dims> shape(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    shape[i] = {Index(jpeg_sizes[i % jpegs.size()])};
  }
  batch->Resize(shape);
    
  // Copy in the data
  batch->template data<uint8>();
  for (int i = 0; i < batch_size; ++i) {
    CUDA_CALL(cudaMemcpy(batch->raw_datum(i),
            jpegs[i % jpegs.size()],
            jpeg_sizes[i & jpegs.size()],
            cudaMemcpyDefault));
  }
  return batch;
}

template <typename T, typename Backend>
void DumpHWCImageBatchToFile(Batch<Backend> &batch) {
  int batch_size = batch.ndatum();
  for (int i = 0; i < batch_size; ++i) {
    vector<Index> shape = batch.datum_shape(i);
    NDLL_ENFORCE(shape.size() == 3);
    int h = shape[0], w = shape[1], c = shape[2];

    DumpHWCToFile((T*)batch.raw_datum(i), h, w, c, w*c, std::to_string(i) + "-batch");
  }
}
  
template <typename T, typename Backend>
void DumpCHWImageBatchToFile(Batch<Backend> &batch) {
  int batch_size = batch.ndatum();
  for (int i = 0; i < batch_size; ++i) {
    vector<Index> shape = batch.datum_shape(i);
    NDLL_ENFORCE(shape.size() == 3);
    int c = shape[0], h = shape[1], w = shape[2];

    DumpCHWToFile((T*)batch.raw_datum(i), h, w, c, std::to_string(i) + "-batch");
  }
    
}

} // namespace ndll

#endif // NDLL_UTIL_IMAGE_H_
