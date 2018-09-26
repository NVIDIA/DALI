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
  ~ImgSetDescr()                                { clear(); }
  inline void clear() {
    for (auto &ptr : data_) delete[] ptr;
    data_.clear(); shapes_.clear();
  }

  inline size_t nImages() const                 { return data_.size(); }
  uint8 *addImage(int imgSize, const uint8 *pRaster = nullptr) {
    shapes_.push_back({imgSize});
    return AddRaster(imgSize, pRaster);
  }

  uint8 *addImage(int h, int w, int c, const uint8 *pRaster = nullptr) {
    shapes_.push_back({h, w, c});
    return AddRaster(h * w * c, pRaster);
  }

  inline void addImageName(const string &name)  { img_names_.push_back(name); }
  const vector<string> &imgNames() const        { return img_names_; }

  void copyImage(int idx, void *pRaster) const  { memcpy(pRaster, data(idx), size(idx)); }

  inline uint8 *data(int idx) const             { return data_[idx]; }
  inline int size(int idx) const                { return shape(idx)[0]; }
  inline const Dims shape(int idx) const        { return shapes_[idx]; }

 /**
 * Loads images from a specified image folder. When imgNames is not defined, assumes the folder
 * contains a file 'image_list.txt' that lists all the different images in the folder
 */
  void LoadImages(const string &image_folder, const vector<string> *imgNames = NULL) {
    if (!imgNames) {
      const string image_list = image_folder + "/image_list.txt";
      std::ifstream file(image_list);
      DALI_ENFORCE(file.is_open());

      string img;
      while (file >> img) {
        DALI_ENFORCE(img.size());
        addImageName(image_folder + "/" + img);
      }
    } else {
      for (auto img : *imgNames) {
        DALI_ENFORCE(img.size());
        addImageName(image_folder + img);
      }
    }

    LoadImages();
  }

  void LoadImages(const vector<string> *imgNames = NULL) {
    if (!imgNames)
      imgNames = &img_names_;

    for (auto img_name : *imgNames) {
      std::ifstream img_file(img_name);
      DALI_ENFORCE(img_file.is_open());

      img_file.seekg(0, std::ios::end);
      int img_size = static_cast<int>(img_file.tellg());
      img_file.seekg(0, std::ios::beg);

      auto data = addImage(img_size);
      img_file.read(reinterpret_cast<char*>(data), img_size);
    }
  }

 private:
  uint8 *AddRaster(int imgSize, const uint8 *pRaster) {
    uint8 *pRasterTo = new uint8[imgSize];
    data_.push_back(pRasterTo);
    if (pRaster)
      memcpy(pRasterTo, pRaster, imgSize);

    return pRasterTo;
  }

  vector<uint8 *> data_;
  vector<Dims> shapes_;
  vector<string> img_names_;
};

/**
 * @brief Writes the input image as a ppm file
 */
DLL_PUBLIC void WriteHWCImage(const uint8 *img, int h, int w, int c, const string &file_name);
DLL_PUBLIC void WriteBatch(const TensorList<CPUBackend> &tl, const string &suffix,
                           float bias = 0.f, float scale = 1.f);

DLL_PUBLIC int idxHWC(int h, int w, int c, int i, int j, int k);
DLL_PUBLIC int idxCHW(int h, int w, int c, int i, int j, int k);

typedef int (*outIdx)(int h, int w, int c, int i, int j, int k);

template <typename T, typename S>
void ConvertRaster(const T *img, int h, int w, int c, vector<S> *pRaster) {
  DALI_ENFORCE(img != nullptr);
  DALI_ENFORCE(h >= 0);
  DALI_ENFORCE(w >= 0);
  DALI_ENFORCE(c >= 0);
  DALI_ENFORCE(pRaster != NULL);

  CUDA_CALL(cudaDeviceSynchronize());
  Tensor<GPUBackend> tmp_gpu, res_gpu;
  tmp_gpu.Resize({h, w, c});
  tmp_gpu.template mutable_data<T>();  // make sure the buffer is allocated
  res_gpu.Resize({h, w, c});

// Copy the data and convert
  MemCopy(tmp_gpu.template mutable_data<T>(), img, tmp_gpu.nbytes());
  Convert(tmp_gpu.template data<T>(), tmp_gpu.size(), res_gpu.template mutable_data<S>());

  pRaster->resize(h * w * c);
  MemCopy(pRaster->data(), res_gpu.template data<S>(), res_gpu.nbytes());
  CUDA_CALL(cudaDeviceSynchronize());
}

/**
 * @brief Writes an image after applying a scale and bias to get
 * pixel values in the range 0-255
 */
template <typename T>
void WriteImageScaleBias(const T *img, int h, int w,
    int c, float bias, float scale, const string &file_name, outIdx pFunc) {
  vector<double> tmp;
  ConvertRaster(img, h, w, c, &tmp);

  std::ofstream file(file_name + ".ppm");
  DALI_ENFORCE(file.is_open());

  file << (c == 3? "P3" : "P2") << endl;    // For color/grayscale images, respectively
  file << w << " " << h << endl;
  file << "255" << endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        const auto idx = (*pFunc)(h, w, c, i, j, k);
        file << static_cast<int>(tmp[idx] * scale + bias) << " ";
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
                const std::array<int, 3> &permute, outIdx pFunc) {
  DALI_ENFORCE(IsType<T>(tl.type()));
  for (int i = 0; i < tl.ntensor(); ++i) {
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
  WriteBatch<T, Backend>(tl, bias, scale, suffix, std::array<int, 3>{0, 1, 2}, idxHWC);
}

template <typename T, typename Backend>
void WriteCHWBatch(const TensorList<Backend> &tl, float bias, float scale, const string &suffix) {
  WriteBatch<T, Backend>(tl, bias, scale, suffix, std::array<int, 3>{1, 2, 0}, idxCHW);
}

template <typename Backend>
void WriteHWCBatch(const TensorList<Backend> &tl, const string &suffix) {
  WriteBatch<uint8, Backend>(tl, 0.f, 1.0, suffix, std::array<int, 3>{0, 1, 2}, idxHWC);
}

}  // namespace dali

#endif  // DALI_UTIL_IMAGE_H_
