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

#ifndef DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_
#define DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_

#include <cmath>
#include "dali/core/force_inline.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T>
struct HtkMelScale {
  DALI_FORCEINLINE T hz_to_mel(T hz) {
    // equivalent to `2595.0 * std::log10(1 + hz / 700.0)`
    return T(1127) * std::log(T(1) + hz / T(700));
  }

  DALI_FORCEINLINE T mel_to_hz(T mel) {
    // equivalent to `700.0 * (std::pow(10, mel / 2595.0) - 1.0)`
    return T(700) * (std::exp(mel / T(1127)) - T(1));
  }
};

template <typename T>
struct SlaneyMelScale {
  static constexpr T freq_low = 0;
  static constexpr T fsp = 200.0 / 3.0;

  static constexpr T min_log_hz = 1000.0;
  static constexpr T min_log_mel = (min_log_hz - freq_low) / fsp;
  static constexpr T step_log = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

  DALI_FORCEINLINE T hz_to_mel(T hz) {
    T mel = 0;
    if (hz >= min_log_hz) {
      // non-linear scale
      mel = min_log_mel + std::log(hz / min_log_hz) / step_log;
    } else {
      // linear scale
      mel = (hz - freq_low) / fsp;
    }

    return mel;
  }

  DALI_FORCEINLINE T mel_to_hz(T mel) {
    T hz = 0;
    if (mel >= min_log_mel) {
      // non linear scale
      hz = min_log_hz * std::exp(step_log * (mel - min_log_mel));
    } else {
      // linear scale
      hz = freq_low + mel * fsp;
    }
    return hz;
  }
};


}  // namespace audio
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_AUDIO_MEL_SCALE_MEL_SCALE_H_
