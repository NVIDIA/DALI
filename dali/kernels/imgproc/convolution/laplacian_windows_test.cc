// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "dali/kernels/common/utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

#include "dali/kernels/imgproc/convolution/laplacian_windows.h"

namespace dali {
namespace kernels {

void CheckDerivWindow(int window_size, LaplacianWindows<float> &windows) {
  cv::Mat d, s;
  cv::getDerivKernels(d, s, 2, 0, window_size, true, CV_32F);
  const auto &window_view = windows.GetDerivWindow(window_size);
  float d_scale = std::exp2f(-window_size + 3);
  for (int i = 0; i < window_size; i++) {
    EXPECT_NEAR(window_view.data[i] * d_scale, d.at<float>(i), 1e-6f)
        << "window_size: " << window_size << ", position: " << i;
  }
}

void CheckSmoothingWindow(int window_size, LaplacianWindows<float> &windows) {
  cv::Mat d, s;
  cv::getDerivKernels(d, s, 2, 0, window_size, true, CV_32F);
  const auto &window_view = windows.GetSmoothingWindow(window_size);
  float s_scale = std::exp2f(-window_size + 1);
  for (int i = 0; i < window_size; i++) {
    EXPECT_NEAR(window_view.data[i] * s_scale, s.at<float>(i), 1e-6f)
        << "window_size: " << window_size << ", position: " << i;
  }
}

TEST(LaplacianWindowsTest, GetDerivWindows) {
  int max_window = 31;
  LaplacianWindows<float> windows{max_window};
  for (int window_size = 3; window_size <= max_window; window_size += 2) {
    CheckDerivWindow(window_size, windows);
  }
}

TEST(LaplacianWindowsTest, GetSmoothingWindows) {
  int max_window = 31;
  LaplacianWindows<float> windows{max_window};
  for (int window_size = 3; window_size <= max_window; window_size += 2) {
    CheckSmoothingWindow(window_size, windows);
  }
}

TEST(LaplacianWindowsTest, CheckPrecomputed) {
  int max_window = 31;
  LaplacianWindows<float> windows{max_window};
  for (int window_size = max_window; window_size >= 3; window_size -= 2) {
    CheckDerivWindow(window_size, windows);
    CheckSmoothingWindow(window_size, windows);
  }
}

}  // namespace kernels
}  // namespace dali
