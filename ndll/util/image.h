#ifndef NDLL_UTIL_IMAGE_H_
#define NDLL_UTIL_IMAGE_H_

#include <fstream>

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"

//
/// This file contains useful image utilities for reading and writing images
//

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

} // namespace ndll

#endif // NDLL_UTIL_IMAGE_H_
