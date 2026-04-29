// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_cpu_kernel.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/image.h"

namespace dali {
namespace kernels {
namespace jpeg {
namespace test {

using testing::dali_extra_path;

inline cv::Mat rgb2bgr(const cv::Mat &img) {
  cv::Mat out;
  cv::cvtColor(img, out, cv::COLOR_RGB2BGR);
  return out;
}

inline cv::Mat bgr2rgb(const cv::Mat &img) {
  return rgb2bgr(img);
}

class JpegDistortionTestCPU : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  using T = uint8_t;
  bool horz_subsample = std::get<0>(GetParam());
  bool vert_subsample = std::get<1>(GetParam());

  void SetUp() override {
    auto paths = ImageList(dali_extra_path() + "/db/single/bmp", {".bmp"}, 3);
    images_.resize(paths.size());
    for (size_t i = 0; i < paths.size(); i++) {
      images_[i] = bgr2rgb(cv::imread(paths[i]));
      ASSERT_FALSE(images_[i].empty()) << "Failed to load " << paths[i];
    }
  }

  void TestQuality(int q) {
    JpegCompressionDistortionCPU kernel;
    int max_abs_error = vert_subsample && horz_subsample ? 80 : 128;
    double max_avg_error = vert_subsample && horz_subsample ? 3 : 10;

    for (size_t i = 0; i < images_.size(); i++) {
      const auto &in_mat = images_[i];
      cv::Mat out_mat(in_mat.rows, in_mat.cols, CV_8UC3);

      TensorShape<3> sh{in_mat.rows, in_mat.cols, 3};
      TensorView<StorageCPU, const uint8_t, 3> in_view{in_mat.data, sh};
      TensorView<StorageCPU, uint8_t, 3> out_view{out_mat.data, sh};

      kernel.RunSample(out_view, in_view, q, horz_subsample, vert_subsample);

      std::vector<uint8_t> encoded;
      cv::imencode(".jpg", rgb2bgr(in_mat), encoded, {cv::IMWRITE_JPEG_QUALITY, q});
      cv::Mat encoded_mat(1, encoded.size(), CV_8UC1, encoded.data());
      cv::Mat out_ref = bgr2rgb(cv::imdecode(encoded_mat, cv::IMREAD_COLOR));

      cv::Mat diff;
      cv::absdiff(out_mat, out_ref, diff);
      double min_val, max_val;
      cv::minMaxLoc(diff, &min_val, &max_val);
      auto mean = cv::mean(diff);

      EXPECT_LE(max_val, max_abs_error)
          << "image " << i << " q=" << q << " hs=" << horz_subsample
          << " vs=" << vert_subsample;
      for (int d = 0; d < 3; d++) {
        EXPECT_LE(mean[d], max_avg_error)
            << "image " << i << " channel " << d << " q=" << q;
      }
    }
  }

  std::vector<cv::Mat> images_;
};

TEST_P(JpegDistortionTestCPU, JpegCompressionDistortion_LowestQuality) {
  this->TestQuality(1);
}

TEST_P(JpegDistortionTestCPU, JpegCompressionDistortion_LowQuality) {
  this->TestQuality(5);
}

TEST_P(JpegDistortionTestCPU, JpegCompressionDistortion_HighQuality) {
  this->TestQuality(95);
}

TEST_P(JpegDistortionTestCPU, JpegCompressionDistortion_HighestQuality) {
  this->TestQuality(100);
}

INSTANTIATE_TEST_SUITE_P(JpegDistortionTestCPU, JpegDistortionTestCPU, ::testing::Combine(
  ::testing::Values(false, true),
  ::testing::Values(false, true)
));  // NOLINT

}  // namespace test
}  // namespace jpeg
}  // namespace kernels
}  // namespace dali
