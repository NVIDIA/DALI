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

#ifndef DALI_KERNELS_SIGNAL_WINDOW_WINDOW_FUNCTIONS_H_
#define DALI_KERNELS_SIGNAL_WINDOW_WINDOW_FUNCTIONS_H_

#include <cmath>
#include "dali/core/span.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
void HannWindow(span<T> output) {
  int N = output.size();
  double a = (2 * M_PI / N);
  for (int t = 0; t < N; t++) {
    double phase = a * (t + 0.5);
    output[t] = static_cast<T>(0.5 * (1.0 - std::cos(phase)));
  }
}

template <typename T>
void HammingWindow(span<T> output, double a0, double a1) {
  int N = output.size();
  double a = (2 * M_PI / N);
  for (int t = 0; t < N; t++) {
    double phase = a * (t + 0.5);
    output[t] = static_cast<T>(a0 - a1 * std::cos(phase));
  }
}

template <typename T>
void HammingWindow(span<T> output, double a0 = 0.53836) {
  HammingWindow(output, a0, 1.0 - a0);
}

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_WINDOW_FUNCTIONS_H_
