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

#ifndef DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_ARGS_H_
#define DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_ARGS_H_

#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"

namespace dali {
namespace kernels {
namespace signal {

enum class Padding {
  None,
  Zero,
  Reflect
};

/**
 * @param length      length of the signal, in samples
 * @param window_size length of the window, in samples
 * @param step        step between consecutive windows, in samples
 * @param centered    if true, assume signal is padded so that all window _centers_ lie inside
 */
DALI_HOST_DEV DALI_FORCEINLINE
constexpr int64_t num_windows(int64_t length, int window_size, int step, bool centered) {
  if (!centered)
    length -= window_size;
  return length / step + 1;
}

struct ExtractWindowsArgs {
  int     window_length = -1;
  int     window_center = -1;
  int     window_step = -1;
  int     axis = -1;
  Padding padding = Padding::Zero;

  DALI_HOST_DEV
  constexpr inline bool operator==(const ExtractWindowsArgs& oth) const {
    return window_length  == oth.window_length &&
           window_center  == oth.window_center &&
           window_step    == oth.window_step &&
           axis           == oth.axis &&
           padding        == oth.padding;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  constexpr bool operator!=(const ExtractWindowsArgs& oth) const {
    return !operator==(oth);
  }

  DALI_HOST_DEV
  constexpr inline int64_t num_windows(int64_t signal_length) const {
    return signal::num_windows(signal_length, window_length, window_step, padding != Padding::None);
  }
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WINDOW_EXTRACT_WINDOWS_ARGS_H_
