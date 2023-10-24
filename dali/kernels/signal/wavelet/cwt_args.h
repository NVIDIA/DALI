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

#ifndef DALI_KERNELS_SIGNAL_WAVELET_CWT_ARGS_H_
#define DALI_KERNELS_SIGNAL_WAVELET_CWT_ARGS_H_

#include <vector>
#include "dali/operators/signal/wavelet/wavelet_name.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T = float>
struct CwtArgs {
  std::vector<T> a;
  dali::DALIWaveletName wavelet;
  std::vector<T> wavelet_args;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_CWT_ARGS_H_
