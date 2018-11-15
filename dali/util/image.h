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

class ImgSetDescr {
 public:
  ~ImgSetDescr()                  { clear(); }
  inline void clear() {
    for (auto &ptr : data_) delete[] ptr;
    data_.clear(); sizes_.clear();
  }

  inline size_t nImages() const   { return data_.size(); }

  vector<uint8 *> data_;
  vector<int> sizes_;
};

/**
 * Load all images from a list of image names. Assumes names contain
 * full path
 */
DLL_PUBLIC void LoadImages(const vector<string> &image_names, ImgSetDescr *imgs);

/**
 * Loads images from a specified image folder. Assumes the folder contains
 * a file 'image_list.txt' that lists all the different images in the
 * folder
 */
DLL_PUBLIC void LoadImages(const string &image_folder, vector<string> *jpeg_names,
                           ImgSetDescr *imgs);

/**
 * Loads jpegs from a specified image folder. Assumes the folder contains
 * a file 'image_list.txt' that lists all the different images in the
 * folder
 */
DLL_PUBLIC void LoadJPEGS(const string &image_folder, vector<string> *jpeg_names,
                          ImgSetDescr *imgs);

/**
 * Loads all jpegs from the list of image names. Assumes names contains
 * full path
 */
DLL_PUBLIC void LoadJPEGS(const vector<string> &jpeg_names, ImgSetDescr *imgs);

/**
 * @brief Writes the input image as a ppm file
 */
DLL_PUBLIC void WriteHWCImage(const uint8 *img, int h, int w, int c, const string &file_name);
DLL_PUBLIC void WriteBatch(const TensorList<CPUBackend> &tl, const string &suffix,
                           float bias = 0.f, float scale = 1.f);

template <typename T>
int outHWCImage(const vector<T> &tmp, int h, int w, int c,
                int i, int j, int k, float bias, float scale) {
  return static_cast<int>(tmp[i*w*c + j*c + k]*scale + bias);
}

template <typename T>
int outCHWImage(const vector<T> &tmp, int h, int w, int c,
                int i, int j, int k, float bias, float scale) {
  return static_cast<int>(tmp[k*h*w + i*w + j]*scale + bias);
}

typedef int (*outFunc)(const vector<double> &tmp, int h, int w, int c,
                       int i, int j, int k, float bias, float scale);

/**
 * @brief Writes an image after applying a scale and bias to get
 * pixel values in the range 0-255
 */
template <typename T>
void WriteImageScaleBias(const T *img, int h, int w,
    int c, float bias, float scale, const string &file_name, outFunc pFunc) {
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

  vector<double> tmp(h * w * c, 0);
  MemCopy(tmp.data(), double_gpu.template data<double>(), double_gpu.nbytes());
  CUDA_CALL(cudaDeviceSynchronize());
  std::ofstream file(file_name + ".ppm");
  DALI_ENFORCE(file.is_open());

  file << (c == 3? "P3" : "P2") << endl;    // For color/grayscale images, respectively
  file << w << " " << h << endl;
  file << "255" << endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << (*pFunc)(tmp, h, w, c, i, j, k, bias, scale) << " ";
      }
    }
    file << endl;
  }
}

/**
 * @brief Writes all images in a batch with a scale and bias
 */
template <typename T, typename Backend>
void WriteBatch(const TensorList<Backend> &tl, float bias, float scale, const string &suffix,
                const std::array<int, 3> &permute, outFunc pFunc) {
  DALI_ENFORCE(IsType<T>(tl.type()));
  for (size_t i = 0; i < tl.ntensor(); ++i) {
    DALI_ENFORCE(tl.tensor_shape(i).size() == 3);
    int h = tl.tensor_shape(i)[permute[0]];
    int w = tl.tensor_shape(i)[permute[1]];
    int c = tl.tensor_shape(i)[permute[2]];
    WriteImageScaleBias(
        tl.template tensor<T>(i),
        h, w, c, bias, scale,
        std::to_string(i) + "-" + suffix, pFunc);
  }
}

template <typename T, typename Backend>
void WriteHWCBatch(const TensorList<Backend> &tl, float bias, float scale, const string &suffix) {
  WriteBatch<T, Backend>(tl, bias, scale, suffix, std::array<int, 3>{0, 1, 2}, outHWCImage);
}

template <typename T, typename Backend>
void WriteCHWBatch(const TensorList<Backend> &tl, float bias, float scale, const string &suffix) {
  WriteBatch<T, Backend>(tl, bias, scale, suffix, std::array<int, 3>{1, 2, 0}, outCHWImage);
}

template <typename Backend>
void WriteHWCBatch(const TensorList<Backend> &tl, const string &suffix) {
  WriteBatch<uint8, Backend>(tl, 0.f, 1.0, suffix, std::array<int, 3>{0, 1, 2}, outHWCImage);
}

}  // namespace dali

#endif  // DALI_UTIL_IMAGE_H_
