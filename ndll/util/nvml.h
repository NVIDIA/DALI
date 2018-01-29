// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_NVML_H_
#define NDLL_UTIL_NVML_H_

#include <nvml.h>

#include <mutex>

#include "ndll/error_handling.h"
#include "ndll/util/nvml_wrap.h"

namespace ndll {
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
  NDLL_CALL(wrapSymbols());
  NDLL_CALL(wrapNvmlInit());
}

/**
 * @brief Sets the CPU affinity for the calling thread
 */
inline void SetCPUAffinity() {
  std::lock_guard<std::mutex> lock(Mutex());
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  nvmlDevice_t device;
  NDLL_CALL(wrapNvmlDeviceGetHandleByIndex(device_idx, &device));
  NDLL_CALL(wrapNvmlDeviceSetCpuAffinity(device));
}

inline void Shutdown() {
  std::lock_guard<std::mutex> lock(Mutex());
  NDLL_CALL(wrapNvmlShutdown());
}

}  // namespace nvml
}  // namespace ndll

#endif  // NDLL_UTIL_NVML_H_
