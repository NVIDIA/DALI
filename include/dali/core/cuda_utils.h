// Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CUDA_UTILS_H_
#define DALI_CORE_CUDA_UTILS_H_

// Device code utilities go into this file.
// For host code utilities, see cuda_rt_utils.h

#include <cuda_runtime.h>  // for __host__ and __device__
#include <malloc.h>
#include <cstdint>
#include <type_traits>
#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"

namespace dali {

// allocator using device-side malloc/free

template <typename T>
struct device_side_allocator {
  static __device__ T *allocate(size_t count) {
    return static_cast<T *>(malloc(count * sizeof(T)));
  }
  static __device__ void deallocate(T *ptr, size_t) {
    ::free(ptr);
  }
};

// moving and perfect forwarding

template <typename T>
constexpr std::remove_reference_t<T> &&
__host__ __device__ cuda_move(T &&t) noexcept {
  return static_cast<std::remove_reference_t<T> &&>(t);
}

template <class T>
__host__ __device__ constexpr T&&
cuda_forward(std::remove_reference_t<T>& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
__host__ __device__ constexpr T&&
cuda_forward(std::remove_reference_t<T>&& t) noexcept {
  return static_cast<T&&>(t);
}

template <typename T>
__host__ __device__ const T &cuda_max(const T &a, const T &b) {
  return b > a ? b : a;
}

template <typename T>
__host__ __device__ const T &cuda_min(const T &a, const T &b) {
  return b < a ? b : a;
}

// swap values using move semantics

template <typename T>
__host__ __device__ void cuda_swap(T &a, T &b) {
  T tmp = cuda_move(a);
  a = cuda_move(b);
  b = cuda_move(tmp);
}

}  // namespace dali

// loading

#ifdef __CUDACC__
__device__ DALI_FORCEINLINE bool __ldg(const bool *ptr) {
  static_assert(sizeof(bool) == 1 || sizeof(bool) == 2 || sizeof(bool) == 4);
  if constexpr (sizeof(bool) == 1)
    return __ldg(reinterpret_cast<const uint8_t *>(ptr));
  else if constexpr (sizeof(bool) == 2)
    return __ldg(reinterpret_cast<const uint16_t *>(ptr));
  else if constexpr (sizeof(bool) == 4)
    return __ldg(reinterpret_cast<const uint32_t *>(ptr));
}
#endif

#endif  // DALI_CORE_CUDA_UTILS_H_
