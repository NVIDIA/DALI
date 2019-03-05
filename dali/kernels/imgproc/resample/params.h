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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_PARAMS_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_PARAMS_H_

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

enum class ResamplingFilterType : uint8_t {
  Nearest,
  Linear,
  Triangular,
  Gaussian,
  Cubic,
  Lanczos3,
};

inline const char * FilterName(ResamplingFilterType type) {
  static const char *names[] = { "NN", "Linear", "Triangular", "Gaussian", "Cubic", "Lanczos3" };
  return names[static_cast<int>(type)];
}

constexpr int KeepOriginalSize = -1;

inline float DefaultFilterRadius(ResamplingFilterType type, float in_size, float out_size) {
  switch (type) {
  case ResamplingFilterType::Triangular:
    return in_size > out_size ? in_size/out_size : 1;
  case ResamplingFilterType::Gaussian:
    return in_size > out_size ? in_size/out_size : 1;
  case ResamplingFilterType::Cubic:
    return 2;
  case ResamplingFilterType::Lanczos3:
    return 3;
  default:
    return 1;
  }
}

struct FilterDesc {
  constexpr FilterDesc() = default;
  constexpr FilterDesc(ResamplingFilterType type, float radius = 0)  // NOLINT
  : type(type), radius(radius) {}
  ResamplingFilterType type = ResamplingFilterType::Nearest;
  float radius = 0;
};

/// @brief Resampling parameters for 1 dimension
struct ResamplingParams {
  FilterDesc min_filter, mag_filter;
  int output_size = KeepOriginalSize;

  /// @brief 1D region of interest
  struct ROI {
    ROI() = default;
    ROI(float start, float end) : use_roi(true), start(start), end(end) {}
    bool use_roi = false;
    float start = 0;
    float end = 0;
  };
  ROI roi;
};

using ResamplingParams2D = std::array<ResamplingParams, 2>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_PARAMS_H_
