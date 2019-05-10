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

#ifndef DALI_CORE_GPU_UTILS_H_
#define DALI_CORE_GPU_UTILS_H_

#include <cuda_runtime.h>
#include "dali/core/cuda_utils.h"

namespace dali {

class DeviceGuard {
 public:
  DeviceGuard() {
    cudaGetDevice(&original_device_);
  }
  explicit DeviceGuard(int new_device) {
    cudaGetDevice(&original_device_);
    CUDA_CALL(cudaSetDevice(new_device));
  }
  ~DeviceGuard() {
    if (cudaSetDevice(original_device_) != cudaSuccess) {
      auto err = cudaGetLastError();
      auto errstr = cudaGetErrorString(err);
      std::cerr << "Failed to recover from DeviceGuard - error " << err
                << ":\n" << errstr << std::endl;
      std::terminate();
    }
  }
 private:
  int original_device_;
};

}  // namespace dali

#endif  // DALI_CORE_GPU_UTILS_H_
