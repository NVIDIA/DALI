// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_UTIL_NVML_H_
#define DALI_UTIL_NVML_H_

#include <nvml.h>

#include <mutex>

#include "dali/error_handling.h"
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

#endif  // DALI_UTIL_NVML_H_
