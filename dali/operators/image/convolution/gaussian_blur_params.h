// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_PARAMS_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_PARAMS_H_

#include <vector>

#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

int GaussianSigmaToDiameter(float sigma) {
  return 2 * ceilf(sigma * 3) + 1;
}

float GaussianDiameterToSigma(int diameter) {
  // Based on OpenCV
  int radius = (diameter - 1) / 2;
  return (radius - 1) * 0.3 + 0.8;
}

struct GaussianDimDesc {
  int usable_axes_start;
  int usable_axes_count;
  bool has_channels;
  bool is_sequence;
};

template <int axes>
struct GaussianSampleParams {
  std::array<int, axes> window_sizes;
  std::array<float, axes> sigmas;

  bool IsUniform() const {
    for (int i = 1; i < axes; i++) {
      if (sigmas[i - 1] != sigmas[i] || window_sizes[i - 1] != window_sizes[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const GaussianSampleParams<axes> &other) const {
    return window_sizes == other.window_sizes && sigmas == other.sigmas;
  }

  bool operator!=(const GaussianSampleParams<axes> &other) const {
    return !(*this == other);
  }
};

void FillGaussian(const TensorView<StorageCPU, float, 1> &window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  // 1 / sqrt(2 * pi * sigma^2) * exp(-(x^2) / (2 * sigma^2))
  // the 1 / sqrt(2 * pi * sigma^2) coefficient disappears as we normalize the sum to 1.
  double exp_scale = 0.5 / (sigma * sigma);
  double sum = 0.;
  // Calculate first half
  for (int x = -r; x < 0; x++) {
    *window(x + r) = exp(-(x * x * exp_scale));
    sum += *window(x + r);
  }
  // Total sum, it's symmetric with `1` in the center.
  sum *= 2.;
  sum += 1.0;
  double scale = 1. / sum;
  // place center, scaled element
  *window(r) = scale;
  // scale all elements so they sum up to 1, duplicate the second half
  for (int x = 0; x < r; x++) {
    *window(x) *= scale;
    *window(2 * r - x) = *window(x);
  }
}

template <int axes>
class GaussianWindows {
 public:
  GaussianWindows() {
    previous.sigmas = uniform_array<axes>(-1.f);
    previous.window_sizes = uniform_array<axes>(0);
  }

  void PrepareWindows(const GaussianSampleParams<axes> &params) {
    bool changed = previous != params;
    if (!changed)
      return;

    // Reallocate if necessary and fill the windows
    bool is_uniform = params.IsUniform();
    if (is_uniform) {
      int required_elements = params.window_sizes[0];
      memory.resize(required_elements);
      TensorView<StorageCPU, float, 1> tmp_view = {memory.data(), {required_elements}};
      FillGaussian(tmp_view, params.sigmas[0]);
      precomputed_window = uniform_array<axes>(
          TensorView<StorageCPU, const float, 1>{memory.data(), {params.window_sizes[0]}});
    } else {
      int required_elements = 0;
      for (int i = 0; i < axes; i++) {
        required_elements += params.window_sizes[i];
      }
      memory.resize(required_elements);
      int offset = 0;
      for (int i = 0; i < axes; i++) {
        TensorView<StorageCPU, float, 1> tmp_view = {&memory[offset], {params.window_sizes[i]}};
        offset += params.window_sizes[i];
        FillGaussian(tmp_view, params.sigmas[i]);
        precomputed_window[i] = tmp_view;
      }
    }
  }

  // Return the already filled windows
  std::array<TensorView<StorageCPU, const float, 1>, axes> GetWindows() {
    return precomputed_window;
  }

 private:
  std::array<TensorView<StorageCPU, const float, 1>, axes> precomputed_window;
  GaussianSampleParams<axes> previous;
  std::vector<float> memory;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_PARAMS_H_
