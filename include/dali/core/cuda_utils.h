// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_fp16.h>  // for __half & related methods
#include <cuda_runtime_api.h>  // for __align__ & CUDART_VERSION
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_error.h"
#include <type_traits>

#if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))
#define DALI_NO_EXEC_CHECK #pragma nv_exec_check_disable
#else
#define DALI_NO_EXEC_CHECK
#endif
#define DALI_HOST_DEV __host__ __device__

// For the CPU we use half_float lib and float16_cpu type
namespace half_float {

class half;

}

namespace dali {

// For the GPU
typedef __half float16;
// For the CPU
typedef half_float::half float16_cpu;

// Compatible wrapper for CUDA 8 which does not have builtin
// static_cast<float16>
template <typename dst>
__device__ inline dst StaticCastGpu(float val) {
  return static_cast<dst>(val);
}

#if defined(__CUDACC__) && defined(CUDART_VERSION) && CUDART_VERSION < 9000
template <>
__device__ inline float16 StaticCastGpu(float val) {
  return __float2half(static_cast<float>(val));
}
#endif  // defined(CUDART_VERSION) && CUDART_VERSION < 9000

// allocator using device-side malloc/free

template <typename T>
struct device_side_allocator {
  static __device__ T *allocate(size_t count) {
    return static_cast<T *>(malloc(count * sizeof(T)));
  }
  static __device__ void deallocate(T *ptr, size_t) {
    free(ptr);
  }
};

// moving and perfect forwwarding

template <typename T>
constexpr typename std::remove_reference<T>::type &&
__host__ __device__ cuda_move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}

template <class T>
__host__ __device__ constexpr T&& cuda_forward(typename std::remove_reference<T>::type& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
__host__ __device__ constexpr T&& cuda_forward(typename std::remove_reference<T>::type&& t) noexcept {
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

}  // namespace dalli

#endif  // DALI_CORE_CUDA_UTILS_H_
