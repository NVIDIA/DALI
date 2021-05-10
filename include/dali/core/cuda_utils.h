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

#include <cuda_runtime_api.h>  // for __align__ & CUDART_VERSION
#include <cassert>
#include <type_traits>
#include "dali/core/host_dev.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_error.h"

namespace dali {

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

/**
 * @brief Gets the maximum number of threads per block for given kernel function on current device
 */
template <typename KernelFunction>
int MaxThreadsPerBlock(KernelFunction *f) {
  static constexpr int kMaxDevices = 1024;
  static int max_block_size[kMaxDevices] = {};
  int device = 0;
  CUDA_CALL(cudaGetDevice(&device));
  assert(device >= 0 && device < kMaxDevices);
  if (!max_block_size[device]) {
    cudaFuncAttributes attr = {};
    CUDA_CALL(cudaFuncGetAttributes(&attr, f));
    max_block_size[device] = attr.maxThreadsPerBlock;
  }
  return max_block_size[device];
}

}  // namespace dali

#endif  // DALI_CORE_CUDA_UTILS_H_
