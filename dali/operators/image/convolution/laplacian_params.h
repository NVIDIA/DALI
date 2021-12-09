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

constexpr static const char *kWindowSizeArgName = "window_size";
constexpr static const char *scaleArgName = "scale";
constexpr static const float scaleArgDefault = 1.0f;
constexpr static const char *normalizeArgName = "normalize";
constexpr static const bool normalizeArgDefault = false;

// OpenCV uses windows sizes up to 31, but may be run on double precision floats.
// For reasonable precision with floats, 23 is set as a maximal window size, because it is
// maximal value for which smoothing window coefficients difference do not exceed 10^6.
// NB. For smoothing window of size 29, float64 and float32 coefficients start to differ.
constexpr static const int maxWindowSize = 23;
constexpr static const int defaultWindowSize = 3;

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
    }
    PrepareSmoothingWindow(window_size - 2);
    PrepareDerivWindow(window_size);
    return deriv_views_[window_idx];
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
      auto prevval = *prev_view(0);
      *view(0) = prevval;
      for (int j = 1; j < cur_size - 1; j++) {
        auto val = *prev_view(j);
        *view(j) = prevval + *prev_view(j);
        prevval = val;
      }
      *view(cur_size - 1) = prevval;
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
      auto prevval = *prev_view(0);
      *view(0) = -prevval;
      for (int j = 1; j < cur_size - 1; j++) {
        auto val = *prev_view(j);
        *view(j) = prevval - *prev_view(j);
        prevval = val;
      }
      *view(cur_size - 1) = prevval;
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
      : spec_{spec},
        has_win_size_t_input_{spec.HasTensorArgument(kWindowSizeArgName)},
        has_scale_t_input_{spec.HasTensorArgument(scaleArgName)},
        has_scale_const_input_{spec.HasArgument(scaleArgName)},
        normalize_{spec.GetArgument<bool>(normalizeArgName)} {}

  void ObtainLaplacianArgs(const ArgumentWorkspace &ws, int nsamples) {
    ObtainWindowSizes(ws, nsamples);
    ObtainScales(ws, nsamples);
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
  void ObtainWindowSizes(const ArgumentWorkspace &ws, int nsamples) {
    if (!has_win_size_t_input_) {
      int prev_size = window_sizes_.size();
      if (prev_size < nsamples) {
        auto window_sizes = prev_size == 0 ? ObtainConstWindowSizes() : window_sizes_[0];
        window_sizes_.resize(nsamples, window_sizes);
      }
    } else {
      window_sizes_.resize(nsamples);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        const auto &tv = ws.ArgumentInput(kWindowSizeArgName);
        const auto &tensor = tv[sample_idx];
        const auto &shape = tensor.shape();
        auto vol = volume(shape);
        DALI_ENFORCE(vol == 1 || vol == axes || vol == num_windows,
                     make_string("Argument `", kWindowSizeArgName, "` for sample ", sample_idx,
                                 " is expected to have 1, ", axes, " or ", axes, "x", axes,
                                 " elements, got ", vol, "."));
        SetSampleWindowSizes(window_sizes_[sample_idx],
                             make_span(tensor.template data<int>(), vol));
        ValidateWindowSizes(window_sizes_[sample_idx], sample_idx);
      }
    }
  }

  std::array<std::array<int, axes>, axes> ObtainConstWindowSizes() {
    std::vector<int> tmp;
    if (!spec_.TryGetRepeatedArgument<int>(tmp, kWindowSizeArgName)) {
      int scalar = spec_.GetArgument<int>(kWindowSizeArgName);
      tmp.assign(axes, scalar);
    }
    std::array<std::array<int, axes>, axes> window_sizes;
    auto in_size = tmp.size();
    DALI_ENFORCE(in_size == 1 || in_size == axes || in_size == num_windows,
                 make_string("Argument `", kWindowSizeArgName, "` is expected to have 1, ", axes,
                             " or ", axes, "x", axes, " elements, got ", in_size, "."));
    SetSampleWindowSizes(window_sizes, make_span(tmp));
    ValidateWindowSizes(window_sizes, 0);
    return window_sizes;
  }

  void ValidateWindowSizes(const std::array<std::array<int, axes>, axes> &window_sizes,
                         int sample_idx) {
    for (int i = 0; i < axes; i++) {
      for (int j = 0; j < axes; j++) {
        bool is_deriv = i == j;
        auto win_size = window_sizes[i][j];
        auto min_size = is_deriv ? 3 : 1;
        DALI_ENFORCE(
            min_size <= win_size && win_size <= maxWindowSize && win_size % 2 == 1,
            make_string((is_deriv ? "Derivative " : "Smoothing "),
                        "window size must be an odd integer between ", min_size, " and ",
                        maxWindowSize, ", got ", win_size, " for sample: ", sample_idx, "."));
      }
    }
  }

  void SetSampleWindowSizes(std::array<std::array<int, axes>, axes> &window_sizes,
                            span<const int> in) {
    auto in_size = in.size();
    if (in_size == num_windows) {
      for (int i = 0; i < axes; i++) {
        memcpy(window_sizes[i].data(), in.data() + i * axes, sizeof(int) * axes);
      }
      return;
    }
    if (in_size == axes) {
      memcpy(window_sizes[0].data(), in.data(), sizeof(int) * axes);
      for (int i = 1; i < axes; i++) {
        for (int j = 0; j < axes; j++) {
          window_sizes[i][j] = window_sizes[0][(axes - i + j) % axes];
        }
      }
      return;
    }
    for (int i = 0; i < axes; i++) {
      for (int j = 0; j < axes; j++) {
        window_sizes[i][j] = in[0];
      }
    }
  }

  void ObtainScales(const ArgumentWorkspace &ws, int nsamples) {
    if (normalize_) {
      DALI_ENFORCE(!has_scale_t_input_ && !has_scale_const_input_,
                   "Parameter ``scale`` cannot be specified when ``normalize`` is set to True");
      if (has_win_size_t_input_) {
        scales_.resize(nsamples);
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
          scales_[sample_idx] = GetNormalizationFactors<axes>(window_sizes_[sample_idx]);
        }
      } else {
        int prev_size = scales_.size();
        if (prev_size < nsamples) {
          auto scale =
              prev_size == 0 ? GetNormalizationFactors<axes>(window_sizes_[0]) : scales_[0];
          scales_.resize(nsamples, scale);
        }
      }
    } else {  // !normalize_
      if (has_scale_t_input_) {
        scales_.resize(nsamples);
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
          GetGeneralizedArg<float>(make_span(scales_[sample_idx]), scaleArgName, sample_idx, spec_,
                                   ws);
        }
      } else {
        int prev_size = scales_.size();
        if (prev_size < nsamples) {
          std::array<float, axes> scale;
          GetGeneralizedArg<float>(make_span(scale), scaleArgName, 0, spec_, ws);
          scales_.resize(nsamples, scale);
        }
      }
    }
  }

  static constexpr int num_windows = axes * axes;
  OpSpec spec_;
  bool has_win_size_t_input_;
  bool has_scale_t_input_;
  bool has_scale_const_input_;

  bool normalize_;
  std::vector<std::array<float, axes>> scales_;
  std::vector<std::array<std::array<int, axes>, axes>> window_sizes_;
};

}  // namespace laplacian
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_PARAMS_H_
