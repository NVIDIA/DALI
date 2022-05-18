// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_H_

#include <cmath>
#include <cassert>
#include <functional>
#include <utility>
#include <vector>
#include "dali/core/math_util.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

struct Args {
  double in_rate = 1, out_rate = 1;
  int64_t out_begin = 0, out_end = -1;  // default values result in the whole range
};

inline double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI));
}

struct ResamplingWindow {
  struct InputRange {
    int i0, i1;
  };

  inline DALI_HOST_DEV InputRange input_range(float x) const {
    int xc = ceilf(x);
    int i0 = xc - lobes;
    int i1 = xc + lobes;
    return {i0, i1};
  }

  /**
   * @brief Calculates the window coefficient at an arbitrary floating point position
   *        by interpolating between two samples.
   */
  DALI_HOST_DEV float operator()(float x) const {
    float fi = x * scale + center;
    float floori = floorf(fi);
    float di = fi - floori;
    int i = floori;
    assert(i >= 0 && i < lookup_size);
    return lookup[i] + di * (lookup[i + 1] - lookup[i]);
  }

  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  int lookup_size = 0;
  const float *lookup = nullptr;
};

struct ResamplingWindowCPU : ResamplingWindow {
  std::vector<float> storage;
};

inline void windowed_sinc(ResamplingWindowCPU &window,
    int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
  assert(coeffs > 1 && lobes > 0 && "Degenerate parameters specified.");
  float scale = 2.0f * lobes / (coeffs - 1);
  float scale_envelope = 2.0f / coeffs;
  window.coeffs = coeffs;
  window.lobes = lobes;
  window.storage.clear();
  window.storage.resize(coeffs + 5);  // add zeros and a full 4-lane vector
  int center = (coeffs - 1) * 0.5f;
  for (int i = 0; i < coeffs; i++) {
    float x = (i - center) * scale;
    float y = (i - center) * scale_envelope;
    float w = sinc(x) * envelope(y);
    window.storage[i + 1] = w;
  }
  window.lookup = window.storage.data();
  window.lookup_size = window.storage.size();
  window.center = center + 1;  // allow for leading zero
  window.scale = 1 / scale;
}

inline int64_t resampled_length(int64_t in_length, double in_rate, double out_rate) {
  return std::ceil(in_length * out_rate / in_rate);
}

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_H_
