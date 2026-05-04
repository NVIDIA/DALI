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
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
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
    // The reference image is produced by cv::imencode(".jpg", ...) which
    // always uses 4:2:0 chroma subsampling. For (horz_subsample,
    // vert_subsample) != (true, true) the kernel produces 4:4:4 / 4:2:2 /
    // 4:4:0 output, so the comparison is necessarily looser -- we assert
    // gross correctness (no crash, no wildly out-of-range pixels) rather
    // than tight numerical agreement. These thresholds mirror the GPU
    // kernel test (jpeg_distortion_gpu_test.cu).
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

// Disabled by default so it doesn't run in normal CI. Invoke with
//   --gtest_filter='*JpegDistortionPerfCPU*' --gtest_also_run_disabled_tests
// to measure RunSample throughput on a synthetic 1080p RGB image.
TEST(JpegDistortionPerfCPU, DISABLED_RunSample_1080p_q75_420) {
  using clock = std::chrono::steady_clock;
  constexpr int H = 1080;
  constexpr int W = 1920;
  constexpr int q = 75;
  constexpr int warmup = 3;
  constexpr int iters = 50;

  std::vector<uint8_t> in(static_cast<size_t>(H) * W * 3);
  std::vector<uint8_t> out(in.size());
  std::mt19937 rng(0xC0FFEE);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &v : in) v = static_cast<uint8_t>(dist(rng));

  TensorShape<3> sh{H, W, 3};
  TensorView<StorageCPU, const uint8_t, 3> in_view{in.data(), sh};
  TensorView<StorageCPU, uint8_t, 3> out_view{out.data(), sh};

  JpegCompressionDistortionCPU kernel;
  for (int i = 0; i < warmup; i++)
    kernel.RunSample(out_view, in_view, q, true, true);

  auto t0 = clock::now();
  for (int i = 0; i < iters; i++)
    kernel.RunSample(out_view, in_view, q, true, true);
  auto t1 = clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double per_call_ms = total_ms / iters;
  double mp = (static_cast<double>(H) * W) / 1e6;
  double mpps = mp * 1000.0 / per_call_ms;

  std::cout << "[ PERF     ] iters=" << iters << " HxW=" << H << "x" << W
            << " q=" << q << " 4:2:0  per-call=" << per_call_ms
            << " ms  total=" << total_ms << " ms  throughput=" << mpps
            << " MP/s" << std::endl;
}

}  // namespace test
}  // namespace jpeg
}  // namespace kernels
}  // namespace dali
