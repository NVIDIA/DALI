// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <opencv2/imgcodecs.hpp>
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/kernels/tensor_shape_print.h"

namespace dali {
namespace kernels {

TEST(ResampleCPU, FilterSymmetry) {
  int w = 9;
  int in_w = 10;
  float scale = (float)in_w / w;
  float sx0 = 0;
  auto filter = GaussianFilter(2);
  int support = filter.support()  ;
  ASSERT_GE(support, 4) << "Gaussian filter with radius 2 must have support of at least 4";
  std::vector<float> coeffs(w * support);
  std::vector<int> idx(w);
  InitializeFilter(idx.data(), coeffs.data(), w, sx0, scale, filter);

  for (int i = 0; i < w / 2; i++) {
    for (int k = 0; k < support; k++) {
      EXPECT_NEAR(coeffs[i*support + k], coeffs[(w-1 - i)*support + support-1 - k], 1e-6f)
        << "Symmetry broken";
    }
  }
  EXPECT_NEAR(coeffs[w/2 * support + support/2 - 1], coeffs[w/2 * support + support/2], 1e-6f)
      << "Central pixel should have symmetrical maximum";

  if (HasFailure()) {
    for (int i = 0; i < w; i++) {
      for (int k = 0; k < support; k++) {
        std::cout << coeffs[i*support + k] << " ";
      }
      std::cout << std::endl;
    }
  }
}

TEST(ResampleCPU, TriangularFilter) {
  int w = 93;
  int in_w = 479;
  float scale = static_cast<float>(in_w) / w;
  float sx0 = 0;
  auto filter = TriangularFilter(scale);
  int support = filter.support();
  ASSERT_EQ(support, 11);
  std::vector<float> coeffs(w * support);
  std::vector<int> idx(w);
  InitializeFilter(idx.data(), coeffs.data(), w, sx0, scale, filter);

  for (int i = 0; i < w; i++) {
    float src = (i + 0.5f) * scale;
    int src_i = floorf(src);
    float max = 0;
    int max_k = 0;
    for (int k = 0; k < support; k++) {
      float v = coeffs[i * support + k];
      if (v > max) {
        max = v;
        max_k = k;
      }
    }
    EXPECT_NEAR(max_k, support/2, 1);
    float slope = coeffs[i*support + 1] - coeffs[i*support + 0];

    // If first (or last) coefficient is larger than the slope, then there should be at least
    // one more contributing sample to the left (or right).
    EXPECT_LT(coeffs[i*support + 0], slope) << "Filter misses a contributing pixel";
    EXPECT_LT(coeffs[i*support + support - 1], slope) << "Filter misses a contributing pixel";
    EXPECT_EQ(idx[i] + max_k, src_i) << "Filter maximum expected to coincide with NN pixel";

  }

}

TEST(ResampleCPU, Horizontal) {
  auto img = testing::data::image("imgproc_test/moire2.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);

  int in_w = img.cols;
  int out_w = 100;

  float scale = static_cast<float>(in_w)/out_w;

  cv::Mat out_img(img.rows, out_w, img.type());
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_img);

  auto filter = TriangularFilter(scale);
  int support = filter.support();
  std::vector<float> coeffs(out_w * support);
  std::vector<int> idx(out_w);
  InitializeFilter(idx.data(), coeffs.data(), out_w, 0, scale, filter);

  ResampleHorz(as_surface_HWC(out_tensor), as_surface_HWC(in_tensor),
    idx.data(), coeffs.data(), support);

  cv::imwrite("horz_300.png", out_img);
}

TEST(ResampleCPU, Vertical) {
  auto img = testing::data::image("imgproc_test/moire2.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);

  int in_h = img.rows;
  int out_h = 100;

  float scale = static_cast<float>(in_h)/out_h;

  cv::Mat out_img(out_h, img.cols, img.type());
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_img);

  auto filter = TriangularFilter(scale);
  int support = filter.support();
  std::vector<float> coeffs(out_h * support);
  std::vector<int> idx(out_h);
  InitializeFilter(idx.data(), coeffs.data(), out_h, 0, scale, filter);

  ResampleVert(as_surface_HWC(out_tensor), as_surface_HWC(in_tensor),
    idx.data(), coeffs.data(), support);

  cv::imwrite("vert_300.png", out_img);
}

}  // namespace kernels
}  // namespace dali
