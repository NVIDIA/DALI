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

#include "dali/core/dev_array.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {
namespace gaussian_blur {

inline int SigmaToDiameter(float sigma) {
  return 2 * ceilf(sigma * 3) + 1;
}

inline float DiameterToSigma(int diameter) {
  // Based on OpenCV
  int radius = (diameter - 1) / 2;
  return (radius - 1) * 0.3 + 0.8;
}

struct DimDesc {
  int usable_axes_start;
  int usable_axes_count;
  int total_axes_count;
  bool has_channels;
  bool is_sequence;
};

template <int axes>
struct GaussianBlurParams {
  DeviceArray<int, axes> window_sizes;
  DeviceArray<float, axes> sigmas;

  bool IsUniform() const {
    for (int i = 1; i < axes; i++) {
      if (sigmas[0] != sigmas[i] || window_sizes[0] != window_sizes[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const GaussianBlurParams<axes> &other) const {
    return window_sizes == other.window_sizes && sigmas == other.sigmas;
  }

  bool operator!=(const GaussianBlurParams<axes> &other) const {
    return !(*this == other);
  }
};

inline void FillGaussian(const TensorView<StorageCPU, float, 1> &window, float sigma) {
  int r = (window.num_elements() - 1) / 2;
  // 1 / sqrt(2 * pi * sigma^2) * exp(-(x^2) / (2 * sigma^2))
  // the 1 / sqrt(2 * pi * sigma^2) coefficient disappears as we normalize the sum to 1.
  float exp_scale = 0.5f / (sigma * sigma);
  float sum = 0.f;
  // Calculate first half
  for (int x = -r; x < 0; x++) {
    *window(x + r) = exp(-(x * x * exp_scale));
    sum += *window(x + r);
  }
  // Total sum, it's symmetric with `1` in the center.
  sum *= 2.;
  sum += 1.0;
  float scale = 1.f / sum;
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

  void PrepareWindows(const GaussianBlurParams<axes> &params) {
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
  std::array<TensorView<StorageCPU, const float, 1>, axes> GetWindows() const {
    return precomputed_window;
  }

 private:
  std::array<TensorView<StorageCPU, const float, 1>, axes> precomputed_window;
  GaussianBlurParams<axes> previous;
  std::vector<float> memory;
};

// This can be fused and we can handle batch of params at a time
// instead of vector of sample params but depending on the parameter changes
// it will probably impact allocation patterns in different ways and need
// to be evaluated if it is fine or not
template <int axes>
void RepackAsTL(std::array<TensorListShape<1>, axes> &out,
                const std::vector<GaussianBlurParams<axes>> &params) {
  for (int axis = 0; axis < axes; axis++) {
    out[axis].resize(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      out[axis].set_tensor_shape(i, {params[i].window_sizes[axis]});
    }
  }
}

template <int axes>
void RepackAsTL(std::array<TensorListView<StorageCPU, const float, 1>, axes> &out,
                const std::vector<GaussianWindows<axes>> &windows) {
  for (int axis = 0; axis < axes; axis++) {
    int nsamples = windows.size();
    out[axis].data.resize(nsamples);
    out[axis].shape.resize(nsamples);
    for (int i = 0; i < nsamples; i++) {
      out[axis].data[i] = windows[i].GetWindows()[axis].data;
      out[axis].shape.set_tensor_shape(i, windows[i].GetWindows()[axis].shape);
    }
  }
}

}  // namespace gaussian_blur
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_PARAMS_H_
