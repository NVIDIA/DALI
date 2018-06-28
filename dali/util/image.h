// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_UTIL_IMAGE_H_
#define DALI_UTIL_IMAGE_H_

#include <cuda_runtime_api.h>

#include <fstream>
#include <vector>
#include <string>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/type_conversion.h"

// This file contains useful image utilities for reading and writing images.
// These functions are for testing and debugging, they should not be used
// inside the library as they throw exceptions on error.

namespace dali {

/**
 * Load all images from a list of image names. Assumes names contain
 * full path
 */
DLL_PUBLIC void LoadImages(const vector<string> &image_names,
    vector<uint8*> *images, vector<int> *image_sizes);

/**
 * Loads images from a specified image folder. Assumes the folder contains
 * a file 'image_list.txt' that lists all the different images in the
 * folder
 */
DLL_PUBLIC void LoadImages(string image_folder, vector<string> *jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes);

/**
 * Loads jpegs from a specified image folder. Assumes the folder contains
 * a file 'image_list.txt' that lists all the different images in the
 * folder
 */
DLL_PUBLIC void LoadJPEGS(string image_folder, vector<string> *jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes);

/**
 * Loads all jpegs from the list of image names. Assumes names contains
 * full path
 */
DLL_PUBLIC void LoadJPEGS(const vector<string> &jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes);

/**
 * @brief Writes the input image as a ppm file
 */
DLL_PUBLIC void WriteHWCImage(const uint8 *img, int h, int w, int c, string file_name);

/**
 * @brief Writes all images in a batch
 */
template <typename Backend>
DLL_PUBLIC void WriteHWCBatch(const TensorList<Backend> &tl, string suffix) {
  DALI_ENFORCE(IsType<uint8>(tl.type()));
  for (int i = 0; i < tl.ntensor(); ++i) {
    DALI_ENFORCE(tl.tensor_shape(i).size() == 3);
    int h = tl.tensor_shape(i)[0];
    int w = tl.tensor_shape(i)[1];
    int c = tl.tensor_shape(i)[2];
    WriteHWCImage(tl.template tensor<uint8>(i),
        h, w, c, std::to_string(i) + "-" + suffix);
  }
}

/**
 * @brief Writes an image after applying a scale and bias to get
 * pixel values in the range 0-255
 */
template <typename T>
void WriteHWCImageScaleBias(const T *img, int h, int w,
    int c, float bias, float scale, string file_name) {
  DALI_ENFORCE(img != nullptr);
  DALI_ENFORCE(h >= 0);
  DALI_ENFORCE(w >= 0);
  DALI_ENFORCE(c >= 0);
  CUDA_CALL(cudaDeviceSynchronize());
  Tensor<GPUBackend> tmp_gpu, double_gpu;
  tmp_gpu.Resize({h, w, c});
  tmp_gpu.template mutable_data<T>();  // make sure the buffer is allocated
  double_gpu.Resize({h, w, c});

  // Copy the data and convert to double
  MemCopy(tmp_gpu.template mutable_data<T>(), img, tmp_gpu.nbytes());
  Convert(tmp_gpu.template data<T>(), tmp_gpu.size(), double_gpu.template mutable_data<double>());

  vector<double> tmp(h*w*c, 0);
  MemCopy(tmp.data(), double_gpu.template data<double>(), double_gpu.nbytes());
  CUDA_CALL(cudaDeviceSynchronize());
  std::ofstream file(file_name + ".ppm");
  DALI_ENFORCE(file.is_open());

  file << "P3" << endl;
  file << w << " " << h << endl;
  file << "255" << endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << int(tmp[i*w*c + j*c + k]*scale + bias) << " ";
      }
    }
    file << endl;
  }
}

/**
 * @brief Writes an image after applying a scale and bias to get
 * pixel values in the range 0-255
 */
template <typename T>
void WriteCHWImageScaleBias(const T *img, int h, int w,
    int c, float bias, float scale, string file_name) {
  DALI_ENFORCE(img != nullptr);
  DALI_ENFORCE(h >= 0);
  DALI_ENFORCE(w >= 0);
  DALI_ENFORCE(c >= 0);
  CUDA_CALL(cudaDeviceSynchronize());
  Tensor<GPUBackend> tmp_gpu, double_gpu;
  tmp_gpu.Resize({h, w, c});
  tmp_gpu.template mutable_data<T>();  // make sure the buffer is allocated
  double_gpu.Resize({h, w, c});

  // Copy the data and convert to double
  MemCopy(tmp_gpu.template mutable_data<T>(), img, tmp_gpu.nbytes());
  Convert(tmp_gpu.template data<T>(), tmp_gpu.size(), double_gpu.template mutable_data<double>());

  vector<double> tmp(h*w*c, 0);
  MemCopy(tmp.data(), double_gpu.template data<double>(), double_gpu.nbytes());
  std::ofstream file(file_name + ".ppm");
  DALI_ENFORCE(file.is_open());

  file << "P3" << endl;
  file << w << " " << h << endl;
  file << "255" << endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << int(tmp[k*h*w + i*w + j]*scale + bias) << " ";
      }
    }
    file << endl;
  }
}

/**
 * @brief Writes all images in a batch with a scale and bias
 */
template <typename T, typename Backend>
void WriteHWCBatch(const TensorList<Backend> &tl, float bias, float scale, string suffix) {
  DALI_ENFORCE(IsType<T>(tl.type()));
  for (int i = 0; i < tl.ntensor(); ++i) {
    DALI_ENFORCE(tl.tensor_shape(i).size() == 3);
    int h = tl.tensor_shape(i)[0];
    int w = tl.tensor_shape(i)[1];
    int c = tl.tensor_shape(i)[2];
    WriteHWCImageScaleBias(
        tl.template tensor<T>(i),
        h, w, c, bias, scale,
        std::to_string(i) + "-" + suffix);
  }
}

/**
 * @brief Writes all images in a batch with a scale and bias
 */
template <typename T, typename Backend>
void WriteCHWBatch(const TensorList<Backend> &tl, float bias, float scale, string suffix) {
  DALI_ENFORCE(IsType<T>(tl.type()));
  for (int i = 0; i < tl.ntensor(); ++i) {
    DALI_ENFORCE(tl.tensor_shape(i).size() == 3);
    int c = tl.tensor_shape(i)[0];
    int h = tl.tensor_shape(i)[1];
    int w = tl.tensor_shape(i)[2];
    WriteCHWImageScaleBias(
        tl.template tensor<T>(i),
        h, w, c, bias, scale,
        std::to_string(i) + "-" + suffix);
  }
}

}  // namespace dali

#endif  // DALI_UTIL_IMAGE_H_
