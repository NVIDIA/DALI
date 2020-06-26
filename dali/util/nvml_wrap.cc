/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ************************************************************************/

#include <dlfcn.h>
#include "dali/util/nvml_wrap.h"
#include "dali/core/cuda_error.h"


namespace dali {

namespace nvml {

int symbolsLoaded = 0;

static nvmlReturn_t (*nvmlInternalInit)(void);
static nvmlReturn_t (*nvmlInternalShutdown)(void);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId,
                                                             nvmlDevice_t* device);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByIndex)(const int device_id,
                                                          nvmlDevice_t* device);
static nvmlReturn_t (*nvmlInternalDeviceGetIndex)(nvmlDevice_t device, unsigned* index);

static nvmlReturn_t (*nvmlInternalDeviceSetCpuAffinity)(nvmlDevice_t device);
static nvmlReturn_t (*nvmlInternalDeviceClearCpuAffinity)(nvmlDevice_t device);
static nvmlReturn_t (*nvmlInternalSystemGetDriverVersion)(char* name, unsigned int length);
static nvmlReturn_t (*nvmlInternalDeviceGetCpuAffinity)(nvmlDevice_t device,
                                                        unsigned int cpuSetSize,
                                                        unsigned long* cpuSet);  // NOLINT(*)

static nvmlReturn_t (*nvmlInternalDeviceGetBrand)(nvmlDevice_t device, nvmlBrandType_t *type);
static nvmlReturn_t (*nvmlInternalDeviceGetCount_v2)(unsigned int *deviceCount);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByIndex_v2)(unsigned int index,
                                                             nvmlDevice_t *device);
static nvmlReturn_t (*nvmlInternalDeviceGetCudaComputeCapability)(nvmlDevice_t device,
                                                                  int *major, int *minor);

static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);

namespace {
bool is_driver_sufficient(int min_cuda_major, int min_cuda_minor) {
  int driver_version;
  CUDA_CALL(cudaDriverGetVersion(&driver_version));
  return driver_version >= 1000 * min_cuda_major + 10 * min_cuda_minor;
}
}  // namespace

bool wrapHasCuda11NvmlFunctions() {
  return nvmlInternalDeviceGetCount_v2 && nvmlInternalDeviceGetHandleByIndex_v2 &&
         nvmlInternalDeviceGetCudaComputeCapability && nvmlInternalDeviceGetBrand;
}

DALIError_t wrapSymbols(void) {
  if (symbolsLoaded)
    return DALISuccess;

  static void* nvmlhandle = NULL;
  void* tmp;
  void** cast;

  nvmlhandle = dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvmlhandle) {
    nvmlhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!nvmlhandle) {
      DALI_FAIL("Failed to open libnvidia-ml.so[.1]");
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {                     \
    cast = reinterpret_cast<void**>(&funcptr);                       \
    tmp = dlsym(handle, symbol);                                     \
    if (tmp == NULL) {                                               \
      DALI_FAIL("dlsym failed on " + symbol + " - " + dlerror());    \
    }                                                                \
    *cast = tmp;                                                     \
  } while (0)

#define LOAD_SYM_MIN_DRIVER(handle, symbol, funcptr, cuda_M, cuda_m) do {    \
    if (!is_driver_sufficient(cuda_M, cuda_m)) {                             \
      funcptr = nullptr;                                                     \
      break;                                                                 \
    }                                                                        \
    cast = reinterpret_cast<void**>(&funcptr);                               \
    tmp = dlsym(handle, symbol);                                             \
    if (tmp == NULL) {                                                       \
      DALI_FAIL("dlsym failed on " + symbol + " - " + dlerror());            \
    }                                                                        \
    *cast = tmp;                                                             \
  } while (0)

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByIndex", nvmlInternalDeviceGetHandleByIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceSetCpuAffinity", nvmlInternalDeviceSetCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlDeviceClearCpuAffinity", nvmlInternalDeviceClearCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlSystemGetDriverVersion", nvmlInternalSystemGetDriverVersion);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetCpuAffinity", nvmlInternalDeviceGetCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);
  LOAD_SYM_MIN_DRIVER(nvmlhandle, "nvmlDeviceGetBrand", nvmlInternalDeviceGetBrand, 11, 0);
  LOAD_SYM_MIN_DRIVER(nvmlhandle, "nvmlDeviceGetCount_v2", nvmlInternalDeviceGetCount_v2, 11, 0);
  LOAD_SYM_MIN_DRIVER(nvmlhandle, "nvmlDeviceGetHandleByIndex_v2",
                      nvmlInternalDeviceGetHandleByIndex_v2, 11, 0);
  LOAD_SYM_MIN_DRIVER(nvmlhandle, "nvmlDeviceGetCudaComputeCapability",
                      nvmlInternalDeviceGetCudaComputeCapability, 11, 0);

  symbolsLoaded = 1;
  return DALISuccess;
}


#define FUNC_BODY(INTERNAL_FUNC, ARGS...)            \
  do {                                               \
    if (INTERNAL_FUNC == NULL) {                     \
      return DALIError;                              \
    }                                                \
    nvmlReturn_t ret = INTERNAL_FUNC(ARGS);          \
    if (ret != NVML_SUCCESS) {                       \
      DALI_WARN(#INTERNAL_FUNC "(...) failed: " +    \
                nvmlInternalErrorString(ret));       \
      return DALIError;                              \
    }                                                \
    return DALISuccess;                              \
  } while (false)


DALIError_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALISuccess;
  }
  nvmlReturn_t ret = nvmlInternalInit();
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlInit() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalShutdown();
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlShutdown() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceGetHandleByPciBusId() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceGetHandleByIndex(const int device_id, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByIndex == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetHandleByIndex(device_id, device);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceGetHandleByIndex() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetIndex(device, index);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceGetIndex() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalDeviceSetCpuAffinity == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceSetCpuAffinity(device);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceSetCpuAffinity() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalInit == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceClearCpuAffinity(device);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceClearCpuAffinity() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlSystemGetDriverVersion(char* name, unsigned int length) {
  if (nvmlInternalInit == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalSystemGetDriverVersion(name, length);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlSystemGetDriverVersion() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceGetCpuAffinity(nvmlDevice_t device,
                                         unsigned int cpuSetSize,
                                         unsigned long* cpuSet) {  // NOLINT(runtime/int)
  if (nvmlInternalDeviceGetCpuAffinity == NULL) {
    DALI_FAIL("lib wrapper not initialized.");
    return DALIError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);
  if (ret != NVML_SUCCESS) {
    DALI_FAIL("nvmlDeviceGetCpuAffinity() failed: " +
      nvmlInternalErrorString(ret));
    return DALIError;
  }
  return DALISuccess;
}

DALIError_t wrapNvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
  FUNC_BODY(nvmlInternalDeviceGetBrand, device, type);
}

DALIError_t wrapNvmlDeviceGetCount_v2(unsigned int* deviceCount) {
  FUNC_BODY(nvmlInternalDeviceGetCount_v2, deviceCount);
}

DALIError_t wrapNvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
  FUNC_BODY(nvmlInternalDeviceGetHandleByIndex_v2, index, device);
}

DALIError_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
  FUNC_BODY(nvmlInternalDeviceGetCudaComputeCapability, device, major, minor);
}

#undef FUNC_BODY


}  // namespace nvml

}  // namespace dali
