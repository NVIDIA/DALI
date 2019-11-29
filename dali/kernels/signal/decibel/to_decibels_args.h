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

#ifndef DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_ARGS_H_
#define DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_ARGS_H_

namespace dali {
namespace kernels {
namespace signal {

template <typename T = float>
struct ToDecibelsArgs {
  // multiplier to the log10 (i.e. `dB = multiplier * log10(s_ratio)`)
  T multiplier = 10.0;

  // reference or denominator of the ratio (i.e `s_ratio = s / s_ref`)
  // If ref_max is set to true, this value will be ignored
  T s_ref = 1.0;

  // cutoff value or minimum value for the s_ratio (smaller values will saturate to this)
  // e.g. a min_ratio of 1e-8 corresponds to a cutoff of -80 in dB
  T min_ratio = 1e-8;

  // If set to true, the maximum of the signal will be used as reference instead of `s_ref`
  bool ref_max = false;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DECIBEL_TO_DECIBELS_ARGS_H_
