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

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>

#include "dali/common.h"
#include "dali/test/dali_test.h"
#include "dali/image/jpeg.h"

namespace dali {

namespace {
// Our turbo jpeg decoder cannot handle CMYK images
// or 410 images
const vector<string> tjpg_test_images = {
  image_folder + "/420.jpg",
  image_folder + "/422.jpg",
  image_folder + "/440.jpg",
  image_folder + "/444.jpg",
  image_folder + "/gray.jpg",
  image_folder + "/411.jpg",
  image_folder + "/411-non-multiple-4-width.jpg",
  image_folder + "/420-odd-height.jpg",
  image_folder + "/420-odd-width.jpg",
  image_folder + "/420-odd-both.jpg",
  image_folder + "/422-odd-width.jpg"
};
}  // namespace

// Fixture for jpeg decode testing. Templated
// to make googletest run our tests grayscale & rgb
template <typename ImgType>
class JpegDecodeTest : public DALITest {
 public:
  void SetUp() {
    if (IsColor(img_type_)) {
      c_ = 3;
    } else if (img_type_ == DALI_GRAY) {
      c_ = 1;
    }
    rand_gen_.seed(time(nullptr));
    LoadJPEGS(tjpg_test_images, &jpegs_, &jpeg_sizes_);
  }

  void TearDown() {
    DALITest::TearDown();
  }

  void VerifyDecode(const uint8 *img, int h, int w, int img_id) {
    // Compare w/ opencv result
    cv::Mat ver;
    cv::Mat jpeg = cv::Mat(1, jpeg_sizes_[img_id], CV_8UC1, jpegs_[img_id]);

    ASSERT_TRUE(CheckIsJPEG(jpegs_[img_id], jpeg_sizes_[img_id]));
    int flag = IsColor(img_type_) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::imdecode(jpeg, flag, &ver);

    cv::Mat ver_img(h, w, IsColor(img_type_) ? CV_8UC3 : CV_8UC2);
    if (img_type_ == DALI_RGB) {
      // Convert from BGR to RGB for verification
      cv::cvtColor(ver, ver_img, CV_BGR2RGB);
    } else {
      ver_img = ver;
    }

    ASSERT_EQ(h, ver_img.rows);
    ASSERT_EQ(w, ver_img.cols);
    vector<int> diff(h*w*c_, 0);
    for (int i = 0; i < h*w*c_; ++i) {
      diff[i] = abs(static_cast<int>(ver_img.ptr()[i] - img[i]));
    }

    // calculate the MSE
    float mean, std;
    MeanStdDev(diff, &mean, &std);

#ifndef NDEBUG
    cout << "num: " << diff.size() << endl;
    cout << "mean: " << mean << endl;
    cout << "std: " << std << endl;
#endif

    // Note: We allow a slight deviation from the ground truth.
    // This value was picked fairly arbitrarily to let the test
    // pass for libjpeg turbo
    ASSERT_LT(mean, 2.f);
    ASSERT_LT(std, 3.f);
  }

  void MeanStdDev(const vector<int> &diff, float *mean, float *std) {
    // Avoid division by zero
    ASSERT_NE(diff.size(), 0);

    double sum = 0, var_sum = 0;
    for (auto &val : diff) {
      sum += val;
    }
    *mean = sum / diff.size();
    for (auto &val : diff) {
      var_sum += (val - *mean)*(val - *mean);
    }
    *std = sqrt(var_sum / diff.size());
  }

 protected:
  const DALIImageType img_type_ = ImgType::type;
  int c_;
};

// Run RGB & grayscale tests
typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(JpegDecodeTest, Types);

TYPED_TEST(JpegDecodeTest, DecodeJPEGHost) {
  vector<uint8> image;
  for (size_t img = 0; img < this->jpegs_.size(); ++img) {
    Tensor<CPUBackend> t;
    DALI_CALL(DecodeJPEGHost(this->jpegs_[img],
            this->jpeg_sizes_[img],
            this->img_type_, &t));
#ifndef NDEBUG
    cout << img << " " << tjpg_test_images[img] << " " << this->jpeg_sizes_[img] << endl;
    cout << "dims: " << t.dim(1) << "x" << t.dim(0) << endl;
#endif
    this->VerifyDecode(t.data<uint8_t>(), t.dim(0), t.dim(1), img);
  }
}

}  // namespace dali
