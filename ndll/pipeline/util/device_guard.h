// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_UTIL_DEVICE_GUARD_H_
#define NDLL_PIPELINE_UTIL_DEVICE_GUARD_H_

#include "ndll/common.h"

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
    CUDA_CALL(cudaSetDevice(original_device_));
  }
 private:
  int original_device_;
};

#endif  // NDLL_PIPELINE_UTIL_DEVICE_GUARD_H_
