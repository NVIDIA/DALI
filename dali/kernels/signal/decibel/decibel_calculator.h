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
#include "dali/core/force_inline.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct MagnitudeToDecibel {
 public:
  DALI_HOST_DEV DALI_FORCEINLINE
  explicit MagnitudeToDecibel(T mul = 10.0, T s_ref = 1.0, T min_ratio = 1e-8)
      : mul_log2_(mul * kLog2Factor)
      , inv_s_ref_(s_ref == T(1) ? T(1) : T(1) / s_ref)
      , min_ratio_(min_ratio) {
    assert(min_ratio_ > T(0));
    assert(inv_s_ref_ > T(0));
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  T operator()(T s) const {
    T ratio = s * inv_s_ref_;
#ifndef __CUDA_ARCH__
    using std::log2;
#endif
    return mul_log2_ * log2(cuda_max(min_ratio_, ratio));
  }

 private:
  static constexpr T kLog2Factor = 0.3010299956639812;  // std::log10(2.0);

  // equivalent multiplier in terms of log2
  T mul_log2_;

  // Inverse of the magnitude reference to which we are calculating the ratio against
  T inv_s_ref_;

  // Cut-off or minimum value for `s/s_ref` ratio
  T min_ratio_;
};

template <typename T>
struct DecibelToMagnitude {
 public:
  DALI_HOST_DEV DALI_FORCEINLINE
  explicit DecibelToMagnitude(T mul = 10.0, T s_ref = 1.0, T min_ratio = 1e-8)
      : inv_mul_(T(1) / mul)
      , s_ref_(s_ref) {}

  DALI_HOST_DEV DALI_FORCEINLINE
  T operator()(T db) const {
#ifndef __CUDA_ARCH__
    using std::pow;
#endif
    return s_ref_ * pow(T(10), db * inv_mul_);
  }

 private:
  static constexpr T kLog2Factor = 0.3010299956639812;  // std::log10(2.0);

  // Inverse of the multiplier
  T inv_mul_;

  // The reference magnitude to which we are calculating the ratio against
  T s_ref_;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DECIBEL_DECIBEL_CALCULATOR_H_
