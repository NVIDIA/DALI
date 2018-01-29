/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


// Dynamically handle dependencies on external libraries (other than cudart).

#ifndef NDLL_UTIL_NVML_WRAP_H_
#define NDLL_UTIL_NVML_WRAP_H_

#include <nvml.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

namespace nvml {

NDLLError_t wrapSymbols(void);

NDLLError_t wrapNvmlInit(void);
NDLLError_t wrapNvmlShutdown(void);
NDLLError_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
NDLLError_t wrapNvmlDeviceGetHandleByIndex(const int device_id, nvmlDevice_t* device);
NDLLError_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index);
NDLLError_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device);
NDLLError_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device);

}  // namespace nvml

}  // namespace ndll

#endif  // NDLL_UTIL_NVML_WRAP_H_
