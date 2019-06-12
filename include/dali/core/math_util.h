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

#ifndef DALI_CORE_MATH_UTIL_H_
#define DALI_CORE_MATH_UTIL_H_

#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <cmath>
#endif
#include "dali/core/host_dev.h"

namespace dali {

inline DALI_HOST_DEV int floor_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rd(x);
#else
  return static_cast<int>(std::floor(x));
#endif
}

inline DALI_HOST_DEV int ceil_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_ru(x);
#else
  return static_cast<int>(std::ceil(x));
#endif
}

inline DALI_HOST_DEV int round_int(float x) {
#ifdef __CUDA_ARCH__
  return __float2int_rn(x);
#else
  return static_cast<int>(std::roundf(x));
#endif
}

template <typename T>
DALI_HOST_DEV constexpr T clamp(const T &value, const T &lo, const T &hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

}  // namespace dali

#endif  // DALI_CORE_MATH_UTIL_H_
