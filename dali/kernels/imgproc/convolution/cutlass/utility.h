// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_UTILITY_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_UTILITY_H_

#include <type_traits>

#include "dali/core/cuda_utils.h"

namespace cutlass {
namespace gemm {

// For inner convolution the input is on the left and convolution kernel window matrix on the right

template <bool InnerConv, typename Input, typename Window>
__host__ __device__
std::enable_if_t<InnerConv, Input&&> select_A(Input&& input, Window&& window) {
  return dali::cuda_forward<Input>(input);
}

template <bool InnerConv, typename Input, typename Window>
__host__ __device__
std::enable_if_t<InnerConv, Window&&> select_B(Input&& input, Window&& window) {
  return dali::cuda_forward<Window>(window);
}


template <bool InnerConv, typename Input, typename Window>
__host__ __device__
std::enable_if_t<!InnerConv, Window&&> select_A(Input&& input, Window&& window) {
  return dali::cuda_forward<Window>(window);
}

template <bool InnerConv, typename Input, typename Window>
__host__ __device__
std::enable_if_t<!InnerConv, Input&&> select_B(Input&& input, Window&& window) {
  return dali::cuda_forward<Input>(input);
}

template <bool InnerConv, typename Input, typename Window>
using select_A_t = std::conditional_t<InnerConv, Input, Window>;

template <bool InnerConv, typename Input, typename Window>
using select_B_t = std::conditional_t<InnerConv, Window, Input>;

}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_UTILITY_H_
