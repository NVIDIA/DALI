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

#ifndef DALI_KERNELS_SIGNAL_DECIBEL_DECIBEL_CALCULATOR_H_
#define DALI_KERNELS_SIGNAL_DECIBEL_DECIBEL_CALCULATOR_H_

#include <cmath>

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct DecibelCalculator {
 public:
  explicit DecibelCalculator(T mul = 10.0, T s_ref = 1.0, T min_ratio = 1e-8)
      : mul_log2_(mul / std::log2(10.0))
      , inv_s_ref_(s_ref == 1.0 ? 1.0 : 1.0 / s_ref)
      , min_ratio_(min_ratio) {
    assert(min_ratio_ > 0.0);
    assert(inv_s_ref_ > 0.0);
  }

  T operator()(T s) const {
    T ratio = s * inv_s_ref_;
    return mul_log2_ * std::log2(std::max(min_ratio_, ratio));
  }

 private:
  // equivalent multiplier in terms of log2
  // (e.g. 10.0 when expressing a power ratio, 20.0 for magnitudes)
  T mul_log2_;

  // Inverse of the magnitude reference to which we are calculating the ratio against
  // (i.e `1.0 / s_ref` where s_ref is the reference magnitude or denominator inside the logarithm)
  T inv_s_ref_;

  // Cut-off or minimum value for `s/s_ref` ratio
  // (e.g. 1e-8 means that the decibel output will saturate to a minimum of -80 dB)
  T min_ratio_;
};


}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DECIBEL_DECIBEL_CALCULATOR_H_
