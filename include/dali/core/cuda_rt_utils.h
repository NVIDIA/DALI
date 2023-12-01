// Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CUDA_RT_UTILS_H_
#define DALI_CORE_CUDA_RT_UTILS_H_

// Host-side utilities go into this file.
// For device code utilities, see cuda_utils.h

#include <cuda_runtime_api.h>  // for __align__ & CUDART_VERSION
#include <cassert>
#include <type_traits>
#include <vector>
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_error.h"

namespace dali {

/**
 * @brief Gets the maximum number of threads per block for given kernel function on current device
 */
template <typename KernelFunction, KernelFunction *f>
int MaxThreadsPerBlockStaticImpl() {
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

/**
 * @brief Gets the maximum number of threads per block for given kernel function on current device
 */
template <typename KernelFunction>
int MaxThreadsPerBlock(KernelFunction *f) {
  cudaFuncAttributes attr = {};
  CUDA_CALL(cudaFuncGetAttributes(&attr, f));
  return attr.maxThreadsPerBlock;
}

#define MaxThreadsPerBlockStatic(func) \
  MaxThreadsPerBlockStaticImpl<std::remove_reference_t<decltype(*func)>, func>()

inline const cudaDeviceProp &GetDeviceProperties(int device_id = -1) {
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  static int dev_count = []() {
    int ndevs = 0;
    CUDA_CALL(cudaGetDeviceCount(&ndevs));
    return ndevs;
  }();
  static std::vector<bool> read(dev_count, false);
  static std::vector<cudaDeviceProp> properties(dev_count);
  if (!read[device_id]) {
    CUDA_CALL(cudaGetDeviceProperties(&properties[device_id], device_id));
    read[device_id] = true;
  }
  return properties[device_id];
}

inline int GetSmCount(int device_id = -1) {
  const auto &props = GetDeviceProperties(device_id);
  return props.multiProcessorCount;
}

inline int GetSharedMemPerBlock(int device_id = -1) {
  const auto &props = GetDeviceProperties(device_id);
  return props.sharedMemPerBlock;
}

}  // namespace dali

#endif  // DALI_CORE_CUDA_RT_UTILS_H_
