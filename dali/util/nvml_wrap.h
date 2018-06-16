/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


// Dynamically handle dependencies on external libraries (other than cudart).

#ifndef DALI_UTIL_NVML_WRAP_H_
#define DALI_UTIL_NVML_WRAP_H_

#include <nvml.h>

#include "dali/common.h"
#include "dali/error_handling.h"

namespace dali {

namespace nvml {

DALIError_t wrapSymbols(void);

DALIError_t wrapNvmlInit(void);
DALIError_t wrapNvmlShutdown(void);
DALIError_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
DALIError_t wrapNvmlDeviceGetHandleByIndex(const int device_id, nvmlDevice_t* device);
DALIError_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index);
DALIError_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device);
DALIError_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device);

}  // namespace nvml

}  // namespace dali

#endif  // DALI_UTIL_NVML_WRAP_H_
