// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_RESAMPLING_CPU_H_
#define DALI_KERNELS_SIGNAL_RESAMPLING_CPU_H_

#include "dali/kernels/signal/resampling.h"
#include "dali/core/api_helper.h"

namespace dali {
namespace kernels {
namespace signal {

namespace resampling {

template <typename Out>
DLL_PUBLIC void ResampleCPUImpl(ResamplingWindow window, Out *__restrict__ out, int64_t out_begin,
                                int64_t out_end, double out_rate, const float *__restrict__ in,
                                int64_t n_in, double in_rate, int num_channels);

struct DLL_PUBLIC ResamplerCPU {
  ResamplingWindowCPU window;

  inline void Initialize(int lobes = 16, int lookup_size = 2048) {
    windowed_sinc(window, lookup_size, lobes);
  }

  /**
   * @brief Resample multi-channel (or single channel) signal and convert to Out
   *
   * Calculates a range of resampled signal.
   * The function can resample a region-of-interest (ROI) of the output, specified by `out_begin` and
   * `out_end`. In this case, the output pointer points to the beginning of the ROI.
   */
  template <typename Out>
  void Resample(Out *__restrict__ out, int64_t out_begin, int64_t out_end, double out_rate,
                const float *__restrict__ in, int64_t n_in, double in_rate, int num_channels) {
    ResampleCPUImpl(window, out, out_begin, out_end, out_rate, in, n_in, in_rate, num_channels);
  }
};

}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_RESAMPLING_CPU_H_
