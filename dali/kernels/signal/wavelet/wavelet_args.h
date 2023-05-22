// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WAVELET_ARGS_H_
#define DALI_KERNELS_SIGNAL_WAVELET_ARGS_H_

#include <variant>
#include <vector>
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct WaveletArgs {
  // mother wavelet name
  WaveletName wavelet = WaveletName::HAAR;

  // wavelet shift parameter
  T b = 0.0;

  // lower limit of wavelet samples
  T begin = -1.0;

  // upper limit of wavelet samples
  T end = 1.0;

  // wavelet sampling rate (samples/s)
  T sampling_rate = 1000.0;
};

template class WaveletArgs<float>;
template class WaveletArgs<double>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_ARGS_H_
