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

#ifndef DALI_TEST_DALI_TEST_H_
#define DALI_TEST_DALI_TEST_H_

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/image/jpeg.h"
#include "dali/pipeline/data/backend.h"
#include "dali/util/image.h"

namespace dali {

// Note: this is setup for the binary to be executed from "build"
const string image_folder = "/data/dali/test/test_images";  // NOLINT

struct DimPair { int h = 0, w = 0; };

// Some useful test 'types'
struct RGB {
  static const DALIImageType type = DALI_RGB;
};
struct BGR {
  static const DALIImageType type = DALI_BGR;
};
struct Gray {
  static const DALIImageType type = DALI_GRAY;
};

// Main testing fixture to provide common functionality across tests
class DALITest : public ::testing::Test {
 public:
  virtual inline void SetUp() {
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(image_folder, &jpeg_names_, &jpegs_, &jpeg_sizes_);
  }

  virtual inline void TearDown() {
    for (auto &ptr : jpegs_) delete[] ptr;
    for (auto &ptr : images_) delete[] ptr;
  }

  inline int RandInt(int a, int b) {
    return std::uniform_int_distribution<>(a, b)(rand_gen_);
  }

  template <typename T>
  inline auto RandReal(int a, int b) -> T {
    return std::uniform_real_distribution<>(a, b)(rand_gen_);
  }

  void DecodeImage(const unsigned char *data, int data_size, int c, int img_type,
                          Tensor<CPUBackend> *out, unsigned char *out_dataPntr = NULL) {
    cv::Mat input(1, data_size, CV_8UC1, const_cast<unsigned char*>(data));

    cv::Mat tmp = cv::imdecode(input, c == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);

    // if RGB needed, permute from BGR
    cv::Mat out_img(tmp.rows, tmp.cols, c != 1 ? CV_8UC3 : CV_8UC1);
    if (img_type == DALI_RGB) {
      // Convert from BGR to RGB for verification
      cv::cvtColor(tmp, out_img, CV_BGR2RGB);
    } else {
      out_img = tmp;
    }

    if (out) {
      out->Resize({tmp.rows, tmp.cols, c});
      out_dataPntr = out->mutable_data<unsigned char>();
    }

    std::memcpy(out_dataPntr, out_img.ptr(), out_img.rows * out_img.cols * c);
  }

  inline void DecodeImages(DALIImageType type, const vector<uint8*>& encoded,
                           const vector<int>& encoded_sizes,
                           vector<uint8*> *images, vector<DimPair> *image_dims) {
    images->resize(encoded.size());
    image_dims->resize(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
      cv::Mat img;
      cv::Mat encode = cv::Mat(1, encoded_sizes[i], CV_8UC1, encoded[i]);

      int flag = IsColor(type) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
      cv::imdecode(encode, flag, &img);

      int h = img.rows;
      int w = img.cols;
      cv::Mat out_img(h, w, IsColor(type) ? CV_8UC3 : CV_8UC1);
      if (type == DALI_RGB) {
        // Convert from BGR to RGB for verification
        cv::cvtColor(img, out_img, CV_BGR2RGB);
      } else {
        out_img = img;
      }

      // Copy the decoded image out & save the dims
      ASSERT_TRUE(out_img.isContinuous());
      c_ = IsColor(type) ? 3 : 1;
      (*images)[i] = new uint8[h*w*c_];
      std::memcpy((*images)[i], out_img.ptr(), h*w*c_);

      (*image_dims)[i].h = h;
      (*image_dims)[i].w = w;
    }
  }

  inline void DecodeJPEGS(DALIImageType type) {
    DecodeImages(type, jpegs_, jpeg_sizes_, &images_, &image_dims_);
  }

  inline void MakeDecodedBatch(int n, TensorList<CPUBackend> *tl,
                               const vector<uint8*> &images,
                               const vector<DimPair> &image_dims,
                               const int c) {
    DALI_ENFORCE(images.size() > 0, "Images must be populated to create batches");
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {image_dims[i % images.size()].h,
                  image_dims[i % images.size()].w,
                  c};
    }
    tl->template mutable_data<uint8>();
    tl->Resize(shape);
    for (int i = 0; i < n; ++i) {
      std::memcpy(tl->template mutable_tensor<uint8>(i),
                  images[i % images.size()],
                  Product(tl->tensor_shape(i)));
    }
  }

  inline void MakeImageBatch(int n, TensorList<CPUBackend> *tl,
                             DALIImageType type = DALI_RGB) {
    if (images_.size() == 0) {
      DecodeJPEGS(type);
    }
    MakeDecodedBatch(n, tl, images_, image_dims_, c_);
  }

  // Make a batch (in TensorList) of arbitrary raw data
  inline void MakeEncodedBatch(TensorList<CPUBackend> *tl, int n,
                        const vector<uint8*> &data,
                        const vector<int> &data_sizes) {
    DALI_ENFORCE(data.size() > 0, "data must be populated to create batches");
    vector<Dims> shape(n);
    for (int i = 0; i < n; ++i) {
      shape[i] = {data_sizes[i % data.size()]};
    }

    tl->template mutable_data<uint8>();
    tl->Resize(shape);

    for (int i = 0; i < n; ++i) {
      std::memcpy(tl->template mutable_tensor<uint8>(i),
          data[i % data.size()],
          data_sizes[i % data.size()]);
    }
  }

  // Make a batch (of vector<Tensor>) of arbitrary raw data
  inline void MakeEncodedBatch(vector<Tensor<CPUBackend>> *t, int n,
                            const vector<uint8*> &data,
                            const vector<int> &data_sizes) {
    DALI_ENFORCE(data.size() > 0, "data must be populated to create batches");

    t->resize(n);
    for (int i = 0; i < n; ++i) {
      auto& ti = t->at(i);
      ti = Tensor<CPUBackend>{};
      ti.Resize({data_sizes[i % data.size()]});
      ti.template mutable_data<uint8>();

      std::memcpy(ti.raw_mutable_data(),
                  data[i % data.size()],
                  data_sizes[i % data.size()]);
    }
  }


  inline void MakeJPEGBatch(TensorList<CPUBackend> *tl, int n) {
    MakeEncodedBatch(tl, n, jpegs_, jpeg_sizes_);
  }

  inline void MakeJPEGBatch(vector<Tensor<CPUBackend>> *t, int n) {
    MakeEncodedBatch(t, n, jpegs_, jpeg_sizes_);
  }

  template <typename T>
  void MeanStdDev(const vector<T> &diff, double *mean, double *std) {
    const size_t N = diff.size();
    // Avoid division by zero
    ASSERT_NE(N, 0);

    double sum = 0, var_sum = 0;
    for (auto &val : diff) {
      sum += val;
    }
    *mean = sum / N;
    for (auto &val : diff) {
      var_sum += (val - *mean)*(val - *mean);
    }
    *std = sqrt(var_sum / N);
  }

  // From OCV example :
  // docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
  cv::Scalar MSSIM(uint8 *a, uint8 *b, int h, int w, int c) {
    cv::Mat i1 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, a);
    cv::Mat i2 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, b);

    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                    // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                    // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);       // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map);  // mssim = average of ssim map
    return mssim;
  }

 protected:
  std::mt19937 rand_gen_;
  vector<string> jpeg_names_;
  vector<uint8*> jpegs_;
  vector<int> jpeg_sizes_;

  // Decoded images
  vector<uint8*> images_;
  vector<DimPair> image_dims_;
  int c_;
};
}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_H_
