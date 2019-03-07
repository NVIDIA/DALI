// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_DEVICE_GUARD_H_
#define DALI_UTIL_DEVICE_GUARD_H_

#include "dali/common.h"
#include "dali/error_handling.h"

namespace dali {
/**
 * Simple RAII device handling:
 * Switch to new device on construction, back to old
 * device on destruction
 */
class DeviceGuard {
 public:
  explicit DeviceGuard(int new_device) {
    CUDA_CALL(cudaGetDevice(&original_device_));
    CUDA_CALL(cudaSetDevice(new_device));
  }


  ~DeviceGuard() {
    auto err = cudaSetDevice(original_device_);
    if (err != cudaSuccess) {
      std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
      std::terminate();
    }
  }


 private:
  int original_device_;
};

}  // namespace dali

#endif  // DALI_UTIL_DEVICE_GUARD_H_
