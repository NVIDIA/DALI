// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_
#define DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

enum class WaveletName {
  HAAR,
  DB,
  SYM,
  COIF,
  BIOR,
  MEY,
  GAUS,
  MEXH,
  MORL,
  CGAU,
  SHAN,
  FBSP,
  CMOR
};

template <typename T>
class MotherWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");

 public:
  MotherWavelet(const WaveletName &name);
  ~MotherWavelet() = default;

  __device__ T (*waveletFunc)(T t, T a, T b);
};

template class MotherWavelet<float>;
template class MotherWavelet<double>;

}  // namespace signal
}  // namespace kernel
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_
