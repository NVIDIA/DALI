// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_NVML_H_
#define NDLL_UTIL_NVML_H_

#include <nvml.h>

#include "ndll/error_handling.h"

namespace ndll {
namespace nvml {

/**
 * @brief Initializes the NVML library
 */
inline void Init() {
  NVML_CALL(nvmlInit());
}

/**
 * @brief Sets the CPU affinity for the calling thread
 */
inline void SetCPUAffinity() {
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  nvmlDevice_t device;
  NVML_CALL(nvmlDeviceGetHandleByIndex(device_idx, &device));
  NVML_CALL(nvmlDeviceSetCpuAffinity(device));
}

inline void Shutdown() {
  NVML_CALL(nvmlShutdown());
}

}  // namespace nvml
}  // namespace ndll

#endif  // NDLL_UTIL_NVML_H_
