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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_WINDOWS_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_WINDOWS_H_

#include <vector>

#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {

template <typename T>
class LaplacianWindows {
 public:
  explicit LaplacianWindows(int max_window_size) : smooth_computed_{1}, deriv_computed_{1} {
    Resize(max_window_size);
    *smoothing_views_[0](0) = 1;
    *deriv_views_[0](0) = 1;
  }

  TensorView<StorageCPU, const T, 1> GetDerivWindow(int window_size) {
    assert(1 <= window_size && window_size <= max_window_size_);
    assert(window_size % 2 == 1);
    auto window_idx = window_size / 2;
    PrepareSmoothingWindow(window_size - 2);
    PrepareDerivWindow(window_size);
    return deriv_views_[window_idx];
  }

  TensorView<StorageCPU, const T, 1> GetSmoothingWindow(int window_size) {
    assert(1 <= window_size && window_size <= max_window_size_);
    assert(window_size % 2 == 1);
    auto window_idx = window_size / 2;
    PrepareSmoothingWindow(window_size);
    return smoothing_views_[window_idx];
  }

 private:
  /**
   * @brief Smoothing window of size 2n + 1 is [1, 2, 1] conv composed with itself n - 1 times
   * so that the window has appropriate size: it boils down to computing binominal coefficients:
   * (1 + 1) ^ (2n).
   */
  inline void PrepareSmoothingWindow(int window_size) {
    for (; smooth_computed_ < window_size; smooth_computed_++) {
      auto cur_size = smooth_computed_ + 1;
      auto cur_idx = cur_size / 2;
      auto &prev_view = smoothing_views_[cur_size % 2 == 0 ? cur_idx - 1 : cur_idx];
      auto &view = smoothing_views_[cur_idx];
      auto prev_val = *prev_view(0);
      *view(0) = prev_val;
      for (int j = 1; j < cur_size - 1; j++) {
        auto val = *prev_view(j);
        *view(j) = prev_val + *prev_view(j);
        prev_val = val;
      }
      *view(cur_size - 1) = prev_val;
    }
  }

  /**
   * @brief Derivative window of size 3 is [1, -2, 1] (which is [1, -1] composed with itself).
   * Bigger windows are convolutions of smoothing windows with [1, -2, 1].
   */
  inline void PrepareDerivWindow(int window_size) {
    for (; deriv_computed_ < window_size; deriv_computed_++) {
      auto cur_size = deriv_computed_ + 1;
      auto cur_idx = cur_size / 2;
      auto &prev_view = cur_size % 2 == 0 ? smoothing_views_[cur_idx - 1] : deriv_views_[cur_idx];
      auto &view = deriv_views_[cur_idx];
      auto prev_val = *prev_view(0);
      *view(0) = -prev_val;
      for (int j = 1; j < cur_size - 1; j++) {
        auto val = *prev_view(j);
        *view(j) = prev_val - *prev_view(j);
        prev_val = val;
      }
      *view(cur_size - 1) = prev_val;
    }
  }

  void Resize(int max_window_size) {
    assert(1 <= max_window_size && max_window_size % 2 == 1);
    max_window_size_ = max_window_size;
    int num_windows = (max_window_size + 1) / 2;
    int num_elements = num_windows * num_windows;
    smoothing_memory_.resize(num_elements);
    deriv_memory_.resize(num_elements);
    smoothing_views_.resize(num_windows);
    deriv_views_.resize(num_windows);
    int offset = 0;
    int window_size = 1;
    for (int i = 0; i < num_windows; i++) {
      smoothing_views_[i] = {&smoothing_memory_[offset], {window_size}};
      deriv_views_[i] = {&deriv_memory_[offset], {window_size}};
      offset += window_size;
      window_size += 2;
    }
  }

  int smooth_computed_, deriv_computed_;
  int max_window_size_;
  std::vector<T> smoothing_memory_;
  std::vector<T> deriv_memory_;
  std::vector<TensorView<StorageCPU, T, 1>> smoothing_views_;
  std::vector<TensorView<StorageCPU, T, 1>> deriv_views_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_WINDOWS_H_
