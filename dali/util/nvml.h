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

#ifndef DALI_UTIL_NVML_H_
#define DALI_UTIL_NVML_H_

#include <nvml.h>

#include <mutex>

#include "dali/error_handling.h"
#include "dali/util/cuda_utils.h"
#include "dali/util/nvml_wrap.h"

namespace dali {
namespace nvml {

/**
 * @brief Getter for the nvml mutex
 */
inline std::mutex& Mutex() {
  static std::mutex mutex;
  return mutex;
}

/**
 * @brief Initializes the NVML library
 */
inline void Init() {
  std::lock_guard<std::mutex> lock(Mutex());
  DALI_CALL(wrapSymbols());
  DALI_CALL(wrapNvmlInit());
}

/**
 * @brief Sets the CPU affinity for the calling thread
 */
inline void SetCPUAffinity() {
  std::lock_guard<std::mutex> lock(Mutex());
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  nvmlDevice_t device;
  DALI_CALL(wrapNvmlDeviceGetHandleByIndex(device_idx, &device));
  DALI_CALL(wrapNvmlDeviceSetCpuAffinity(device));
}

inline void Shutdown() {
  std::lock_guard<std::mutex> lock(Mutex());
  DALI_CALL(wrapNvmlShutdown());
}

}  // namespace nvml
}  // namespace dali

#define NVML_CALL(code)                                    \
  do {                                                     \
    nvmlReturn_t status = code;                            \
    if (status != NVML_SUCCESS) {                          \
      dali::string error = dali::string("NVML error \"") + \
        nvmlErrorString(status) + "\"";                    \
      DALI_FAIL(error);                                    \
    }                                                      \
  } while (0)

#endif  // DALI_UTIL_NVML_H_
