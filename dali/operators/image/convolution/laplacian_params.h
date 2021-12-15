// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_PARAMS_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_PARAMS_H_

#include <numeric>
#include <vector>

#include "dali/core/small_vector.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"


namespace dali {
namespace laplacian {

constexpr static const char *windowSizeArgName = "window_size";
constexpr static const int defaultWindowSize = 3;
constexpr static const char *smoothingSizeArgName = "smoothing_size";
constexpr static auto smoothingSizeDefault = nullptr;
constexpr static const char *scaleArgName = "scale";
constexpr static const float scaleArgDefault = 1.0f;
constexpr static const char *normalizeArgName = "normalize";
constexpr static const bool normalizeArgDefault = false;

// The operator uses the same windows as OpenCV. OpenCV uses windows sizes up to 31,
// but it may be run on double precision floats, while DALI supports only
// float32 as intermediate type.
// For reasonable precision with floats, 23 is set as a maximal window size, because it is
// maximal value for which smoothing window coefficients difference do not exceed 10^6.
// NB. For smoothing window of size 29, float64 and float32 coefficients start to differ.
constexpr static const int maxWindowSize = 23;

template <typename T>
class LaplacianWindows {
 public:
  LaplacianWindows()
      : smooth_computed_{1}, deriv_computed_{1}, smoothing_memory_{}, deriv_memory_{} {
    int offset = 0;
    int window_size = 1;
    for (int i = 0; i < num_windows_; i++) {
      smoothing_views_[i] = {&smoothing_memory_[offset], {window_size}};
      deriv_views_[i] = {&deriv_memory_[offset], {window_size}};
      offset += window_size;
      window_size += 2;
    }
    *smoothing_views_[0](0) = 1;
    *deriv_views_[0](0) = 1;
  }

  TensorView<StorageCPU, const T, 1> GetWindow(int window_size, bool is_deriv) {
    assert(1 <= window_size && window_size <= maxWindowSize);
    assert(window_size % 2 == 1);
    auto window_idx = window_size / 2;
    if (!is_deriv) {
      PrepareSmoothingWindow(window_size);
      return smoothing_views_[window_idx];
    } else {
      PrepareSmoothingWindow(window_size - 2);
      PrepareDerivWindow(window_size);
      return deriv_views_[window_idx];
    }
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

  static constexpr int num_windows_ = (maxWindowSize + 1) / 2;
  static constexpr int windows_size_ = (1 + maxWindowSize) / 2 * num_windows_;
  int smooth_computed_, deriv_computed_;
  std::array<T, windows_size_> smoothing_memory_;
  std::array<T, windows_size_> deriv_memory_;
  std::array<TensorView<StorageCPU, T, 1>, num_windows_> smoothing_views_;
  std::array<TensorView<StorageCPU, T, 1>, num_windows_> deriv_views_;
};


template <int axes>
std::array<float, axes> GetNormalizationFactors(
    const std::array<std::array<int, axes>, axes> &window_sizes) {
  std::array<float, axes> factors;
  for (int i = 0; i < axes; i++) {
    auto initial = -axes - 2;
    auto exponent = std::accumulate(window_sizes[i].begin(), window_sizes[i].end(), initial);
    factors[i] = exp2(-exponent);
  }
  return factors;
}


template <int axes>
class LaplacianArgs {
 public:
  explicit LaplacianArgs(const OpSpec &spec)
      : has_deriv_tensor_(spec.HasTensorArgument(windowSizeArgName)),
        has_smooth_const_(spec.HasArgument(smoothingSizeArgName)),
        has_smooth_tensor_(spec.HasTensorArgument(smoothingSizeArgName)),
        has_scales_const_(spec.HasArgument(scaleArgName)),
        has_scales_tensor_(spec.HasTensorArgument(scaleArgName)),
        normalize_{spec.GetArgument<bool>(normalizeArgName)} {}

  void ObtainLaplacianArgs(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples) {
    ObtainWindowSizes(spec, ws, nsamples);
    if (!normalize_) {
      ObtainScales(spec, ws, nsamples);
    } else {
      DALI_ENFORCE(!HasScaleSpecified(),
                   "Parameter ``scale`` cannot be specified when ``normalize`` is set to True");
      ComputeScales(nsamples);
    }
  }

  inline const std::array<std::array<int, axes>, axes> &GetWindowSizes(int sample_idx) {
    return window_sizes_[sample_idx];
  }

  inline const std::array<float, axes> &GetScales(int sample_idx) {
    return scales_[sample_idx];
  }

  inline int GetTotalWindowSizes(int sample_idx) {
    auto &window_sizes = GetWindowSizes(sample_idx);
    auto acc_sizes = [](int acc, const std::array<int, axes> &win_sizes) {
      return std::accumulate(win_sizes.begin(), win_sizes.end(), acc);
    };
    return std::accumulate(window_sizes.begin(), window_sizes.end(), 0, acc_sizes);
  }

 private:
  bool HasPerSampleWindows() {
    return has_deriv_tensor_ || has_smooth_tensor_;
  }

  bool HasSmoothingSpecified() {
    return has_smooth_const_ || has_smooth_tensor_;
  }

  bool HasScaleSpecified() {
    return has_scales_const_ || has_scales_tensor_;
  }

  void ObtainWindowSizes(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples) {
    int prev_size = window_sizes_.size();
    if (HasPerSampleWindows() || prev_size < nsamples) {
      window_sizes_.resize(nsamples);
    }
    int sample_idx = HasPerSampleWindows() ? 0 : prev_size;
    for (; sample_idx < nsamples; sample_idx++) {
      SetSampleWindowSizes(spec, ws, sample_idx);
    }
  }

  void SetSampleWindowSizes(const OpSpec &spec, const ArgumentWorkspace &ws, int sample_idx) {
    auto &window_sizes = window_sizes_[sample_idx];
    std::array<int, axes> deriv_arg;
    GetGeneralizedArg<int>(make_span(deriv_arg), windowSizeArgName, sample_idx, spec, ws);
    for (auto d_size : deriv_arg) {
      DALI_ENFORCE(3 <= d_size && d_size <= maxWindowSize && d_size % 2 == 1,
                   make_string("Window size must be an odd integer between 3 and ", maxWindowSize,
                               ", got ", d_size, " for sample: ", sample_idx, "."));
    }
    if (!HasSmoothingSpecified()) {
      for (int i = 0; i < axes; i++) {
        std::fill(window_sizes[i].begin(), window_sizes[i].end(), deriv_arg[i]);
      }
    } else {
      std::array<int, axes> smooth_arg;
      GetGeneralizedArg<int>(make_span(smooth_arg), smoothingSizeArgName, sample_idx, spec, ws);
      for (auto s_size : smooth_arg) {
        DALI_ENFORCE(
            1 <= s_size && s_size <= maxWindowSize && s_size % 2 == 1,
            make_string("Smoothing window size must be an odd integer between 1 and ",
                        maxWindowSize, ", got ", s_size, " for sample: ", sample_idx, "."));
      }
      for (int i = 0; i < axes; i++) {
        for (int j = 0; j < axes; j++) {
          window_sizes[i][j] = i == j ? deriv_arg[i] : smooth_arg[j];
        }
      }
    }
  }

  void ComputeScales(int nsamples) {
    if (HasPerSampleWindows()) {
      scales_.resize(nsamples);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        scales_[sample_idx] = GetNormalizationFactors<axes>(window_sizes_[sample_idx]);
      }
    } else {
      int prev_size = scales_.size();
      if (prev_size < nsamples) {
        auto scale = prev_size == 0 ? GetNormalizationFactors<axes>(window_sizes_[0]) : scales_[0];
        scales_.resize(nsamples, scale);
      }
    }
  }

  void ObtainScales(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples) {
    int prev_size = scales_.size();
    if (has_scales_tensor_ || prev_size < nsamples) {
      scales_.resize(nsamples);
    }
    for (int sample_idx = has_scales_tensor_ ? 0 : prev_size; sample_idx < nsamples; sample_idx++) {
      GetGeneralizedArg<float>(make_span(scales_[sample_idx]), scaleArgName, sample_idx, spec, ws);
    }
  }

  static constexpr int num_windows = axes * axes;

  bool has_deriv_tensor_;
  bool has_smooth_const_, has_smooth_tensor_;
  bool has_scales_const_, has_scales_tensor_;
  bool normalize_;
  std::vector<std::array<float, axes>> scales_;
  std::vector<std::array<std::array<int, axes>, axes>> window_sizes_;
};

}  // namespace laplacian
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_PARAMS_H_
