#ifndef NDLL_UTIL_IMAGE_H_
#define NDLL_UTIL_IMAGE_H_

#include <fstream>

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/types.h"
#include "ndll/util/type_conversion.h"

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
 * Loads all jpegs from the list of image names. Assumes names contains
 * full path
 */
void LoadJPEGS(const vector<string> &jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes);

/**
 * Writes an HWC image to the specified file. Add the file extension '.txt'
 */
template <typename T>
void DumpHWCToFile(const T *img, int h, int w, int c, string file_name) {
  NDLL_ENFORCE(img != nullptr);
  NDLL_ENFORCE(h >= 0);
  NDLL_ENFORCE(w >= 0);
  NDLL_ENFORCE(c >= 0);
  CUDA_CALL(cudaDeviceSynchronize());
  Tensor<GPUBackend> tmp_gpu, double_gpu;
  tmp_gpu.Resize({h, w, c});
  tmp_gpu.template mutable_data<T>(); // make sure the buffer is allocated
  double_gpu.Resize({h, w, c});

  // Copy the data and convert to double
  MemCopy(tmp_gpu.template mutable_data<T>(), img, tmp_gpu.nbytes());
  Convert(tmp_gpu.template data<T>(), tmp_gpu.size(), double_gpu.template mutable_data<double>());

  vector<double> tmp(h*w*c, 0);
  MemCopy(tmp.data(), double_gpu.template data<double>(), double_gpu.nbytes());
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
}

/**
 * Writes an CHW image to the specified file. Add the file extension '.txt'.
 * The data will be written in HWC format
 */
template <typename T>
void DumpCHWToFile(const T *img, int h, int w, int c, string file_name) {
  CUDA_CALL(cudaDeviceSynchronize());
  Tensor<GPUBackend> tmp_gpu, double_gpu;
  tmp_gpu.Resize({c, h, w});
  tmp_gpu.template mutable_data<T>(); // make sure the buffer is allocated
  double_gpu.Resize({c, h, w});

  // Copy the data and convert to double
  MemCopy(tmp_gpu.template mutable_data<T>(), img, tmp_gpu.nbytes());
  Convert(tmp_gpu.template data<T>(), tmp_gpu.size(), double_gpu.template mutable_data<double>());

  vector<double> tmp(c*h*w, 0);
  MemCopy(tmp.data(), double_gpu.template data<double>(), double_gpu.nbytes());

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
  batch->template mutable_data<uint8>();
  for (int i = 0; i < batch_size; ++i) {
    CUDA_CALL(cudaMemcpy(batch->raw_mutable_sample(i),
            jpegs[i % jpegs.size()],
            jpeg_sizes[i % jpegs.size()],
            cudaMemcpyDefault));
  }
  return batch;
}

template <typename T, typename Backend>
void DumpHWCImageBatchToFile(const Batch<Backend> &batch, const string suffix = "-batch") {
  NDLL_ENFORCE(IsType<T>(batch.type()));
  
  int batch_size = batch.nsample();
  for (int i = 0; i < batch_size; ++i) {
    vector<Index> shape = batch.sample_shape(i);
    NDLL_ENFORCE(shape.size() == 3);
    int h = shape[0], w = shape[1], c = shape[2];

    DumpHWCToFile(batch.template sample<T>(i), h, w, c, std::to_string(i) + suffix);
  }
}
  
template <typename T, typename Backend>
void DumpCHWImageBatchToFile(const Batch<Backend> &batch, const string suffix = "-batch") {
  NDLL_ENFORCE(IsType<T>(batch.type()));
  
  int batch_size = batch.nsample();
  for (int i = 0; i < batch_size; ++i) {
    vector<Index> shape = batch.sample_shape(i);
    NDLL_ENFORCE(shape.size() == 3);
    int c = shape[0], h = shape[1], w = shape[2];

    DumpCHWToFile(batch.template sample<T>(i), h, w, c, std::to_string(i) + suffix);
  }   
}

template <typename T>
void DumpHWCRawImageBatchToFile(T *ptr, int n, int h, int w, int c, const string suffix = "-batch") {
  for (int i = 0; i < n; ++i) {
    DumpHWCToFile(ptr + i*c*h*w,
        h, w, c, std::to_string(i) + suffix);
  }
}

template <typename T>
void DumpCHWRawImageBatchToFile(T *ptr, int n, int h, int w, int c, const string suffix = "-batch") {
  for (int i = 0; i < n; ++i) {
    DumpCHWToFile(ptr + i*c*h*w, h, w, c, std::to_string(i) + suffix);
  }
}

} // namespace ndll

#endif // NDLL_UTIL_IMAGE_H_
