// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

#include "dali/operators/image/convolution/gaussian_blur_params.h"

namespace dali {
namespace gaussian_blur {

// TODO(klecki): OpenCV have special cases of precomputed values for for kernels of sizes 1-7 and
// sigma = 0
TEST(GaussianBlurTest, FillGaussian) {
  std::vector<std::pair<int, float>> size_sigma_pairs = {
      {1, 0},     {3, 0},    {5, 0},      {7, 0},     {9, 0},    {11, 0},    {13, 0},
      {15, 0},    {101, 0},  {0, 0.025f}, {0, 0.25f}, {0, 0.5f}, {0, 0.75f}, {0, 1.f},
      {0, 1.25f}, {0, 1.5f}, {0, 2.f},    {0, 3.f},   {0, 5.f},  {0, 16.f}};
  kernels::TestTensorList<float, 1> window;
  for (const auto &size_sigma : size_sigma_pairs) {
    int size;
    float sigma;
    std::tie(size, sigma) = size_sigma;
    if (size == 0) {
      size = SigmaToDiameter(sigma);
    } else if (sigma == 0.f) {
      sigma = DiameterToSigma(size);
    }
    TensorListShape<1> shape({TensorShape<1>{size}});
    window.reshape(shape);
    auto window_view = window.cpu()[0];
    FillGaussian(window_view, sigma);
    auto mat = cv::getGaussianKernel(size, sigma, CV_32F);
    for (int i = 0; i < size; i++) {
      EXPECT_NEAR(window_view.data[i], mat.at<float>(i), 1e-7f)
          << "size: " << size << ", sigma: " << sigma;
    }
  }
}

template <size_t axes>
void CheckUniformWindows(const std::array<TensorView<StorageCPU, const float, 1>, axes> &views) {
  for (size_t i = 1; i < axes; i++) {
    EXPECT_EQ(views[i - 1].data, views[i].data);
    EXPECT_EQ(views[i - 1].shape, views[i].shape);
  }
}

template <size_t axes>
void CheckNonUniformWindows(const std::array<TensorView<StorageCPU, const float, 1>, axes> &views) {
  for (size_t i = 0; i < axes; i++) {
    for (size_t j = i + 1; j < axes; j++) {
      EXPECT_NE(views[i].data, views[j].data);
    }
  }
}

template <size_t axes, int axes_params = axes>
void CheckMatchingShapes(const std::array<TensorView<StorageCPU, const float, 1>, axes> &views,
                         const GaussianBlurParams<axes_params> &params) {
  for (size_t i = 0; i < axes; i++) {
    EXPECT_EQ(params.window_sizes[i], views[i].shape[0]);
  }
}

template <int axes, size_t num_iters>
auto CollectWindowsForParams(
    std::array<GaussianBlurParams<axes>, num_iters> &params) {
  std::array<std::array<std::vector<float>, axes>, num_iters> windows_hist;
  GaussianWindows<axes> windows;
  for (size_t i = 0; i < num_iters; i++) {
    windows.PrepareWindows(params[i]);
    auto window_views = windows.GetWindows();
    for (int axis = 0; axis < axes; axis++) {
      auto *data = window_views[axis].data;
      windows_hist[i][axis].assign(data, data + window_views[axis].num_elements());
      TensorShape<1> expected_shape = {params[i].window_sizes[axis]};
      EXPECT_EQ(window_views[axis].shape, expected_shape);
    }
  }
  return windows_hist;
}

TEST(GaussianWindowsTest, InitUniform) {
  constexpr int axes = 2;
  GaussianWindows<axes> windows;
  GaussianBlurParams<axes> params_uniform = {{7, 7}, {1.0f, 1.0f}};
  windows.PrepareWindows(params_uniform);
  auto views = windows.GetWindows();
  CheckUniformWindows(views);
  CheckMatchingShapes(views, params_uniform);
}

TEST(GaussianWindowsTest, InitNonUniformSigma) {
  constexpr int axes = 2;
  GaussianWindows<axes> windows;
  GaussianBlurParams<axes> params = {{7, 7}, {1.0f, 2.0f}};
  windows.PrepareWindows(params);
  auto views = windows.GetWindows();
  CheckNonUniformWindows(views);
  CheckMatchingShapes(views, params);
}

TEST(GaussianWindowsTest, InitNonUniformShape) {
  constexpr int axes = 2;
  GaussianWindows<axes> windows;
  GaussianBlurParams<axes> params = {{7, 9}, {1.0f, 1.0f}};
  windows.PrepareWindows(params);
  auto views = windows.GetWindows();
  CheckNonUniformWindows(views);
  CheckMatchingShapes(views, params);
}

TEST(GaussianWindowsTest, Uniform2NonUniform) {
  constexpr int axes = 2;
  GaussianWindows<axes> windows;
  GaussianBlurParams<axes> params_uniform = {{7, 7}, {1.0f, 1.0f}};
  GaussianBlurParams<axes> params_non_uniform = {{7, 14}, {2.0f, 1.0f}};

  windows.PrepareWindows(params_uniform);
  auto views_uniform = windows.GetWindows();
  CheckUniformWindows(views_uniform);
  CheckMatchingShapes(views_uniform, params_uniform);

  windows.PrepareWindows(params_non_uniform);
  auto views_non_uniform = windows.GetWindows();
  CheckNonUniformWindows(views_non_uniform);
  CheckMatchingShapes(views_non_uniform, params_non_uniform);
}

TEST(GaussianWindowsTest, NonUniform2Uniform) {
  constexpr int axes = 2;
  GaussianWindows<axes> windows;
  GaussianBlurParams<axes> params_uniform = {{7, 7}, {1.0f, 1.0f}};
  GaussianBlurParams<axes> params_non_uniform = {{7, 14}, {2.0f, 1.0f}};

  windows.PrepareWindows(params_non_uniform);
  auto views_non_uniform = windows.GetWindows();
  CheckNonUniformWindows(views_non_uniform);
  CheckMatchingShapes(views_non_uniform, params_non_uniform);

  windows.PrepareWindows(params_uniform);
  auto views_uniform = windows.GetWindows();
  CheckUniformWindows(views_uniform);
  CheckMatchingShapes(views_uniform, params_uniform);
}

TEST(GaussianWindowsTest, NoChangeUniform) {
  constexpr int axes = 2;
  constexpr size_t num_iters = 3;
  std::array<GaussianBlurParams<axes>, num_iters> params_uniform = {{
    {{7, 7}, {1.0f, 1.0f}},
    {{7, 7}, {1.0f, 1.0f}},
    {{7, 7}, {1.0f, 1.0f}}
  }};
  auto windows_hist = CollectWindowsForParams(params_uniform);
  for (size_t i = 1; i < num_iters; i++) {
    EXPECT_EQ(windows_hist[i - 1], windows_hist[i]);
  }
}

TEST(GaussianWindowsTest, NoChangeNonUniform) {
  constexpr int axes = 2;
  constexpr size_t num_iters = 3;
  std::array<GaussianBlurParams<axes>, num_iters> params_non_uniform = {{
    {{7, 14}, {2.0f, 1.0f}},
    {{7, 14}, {2.0f, 1.0f}},
    {{7, 14}, {2.0f, 1.0f}}
  }};
  auto windows_hist = CollectWindowsForParams(params_non_uniform);
  for (size_t i = 1; i < num_iters; i++) {
    EXPECT_EQ(windows_hist[i - 1], windows_hist[i]);
  }
}

TEST(GaussianWindowsTest, UpdateWindowsOnSigmaChange) {
  constexpr int axes = 2;
  constexpr size_t num_iters = 4;
  std::array<GaussianBlurParams<axes>, num_iters> params = {{
    {{7, 7}, {1.0f, 1.0f}},
    {{7, 7}, {2.0f, 3.0f}},
    {{7, 7}, {1.0f, 1.0f}},
    {{7, 7}, {1.0f, 2.0f}}
  }};
  auto windows_hist = CollectWindowsForParams(params);
  for (int axis = 0; axis < axes; axis++) {
    EXPECT_NE(windows_hist[0][axis], windows_hist[1][axis]);
    EXPECT_EQ(windows_hist[0][axis], windows_hist[2][axis]);
  }
  EXPECT_EQ(windows_hist[0][0], windows_hist[3][0]);
  EXPECT_EQ(windows_hist[1][0], windows_hist[3][1]);
}

TEST(GaussianWindowsTest, UpdateWindowsOnWindowSizeChange) {
  constexpr int axes = 2;
  constexpr size_t num_iters = 3;
  std::array<GaussianBlurParams<axes>, num_iters> params = {{
    {{7, 7}, {1.0f, 1.0f}},
    {{5, 5}, {1.0f, 1.0f}},
    {{7, 7}, {1.0f, 1.0f}}
  }};
  auto windows_hist = CollectWindowsForParams(params);
  for (int axis = 0; axis < axes; axis++) {
    EXPECT_NE(windows_hist[0][axis], windows_hist[1][axis]);
    EXPECT_EQ(windows_hist[0][axis], windows_hist[2][axis]);
  }
}

}  // namespace gaussian_blur
}  // namespace dali
