/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ndll/util/nvml_wrap.h"

#include <dlfcn.h>

namespace ndll {

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
static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);

NDLLError_t wrapSymbols(void) {
  if (symbolsLoaded)
    return NDLLSuccess;

  static void* nvmlhandle = NULL;
  void* tmp;
  void** cast;

  nvmlhandle = dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvmlhandle) {
    nvmlhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!nvmlhandle) {
      NDLL_FAIL("Failed to open libnvidia-ml.so[.1]");
      goto teardown;
    }
  }

  #define LOAD_SYM(handle, symbol, funcptr) do {                     \
    cast = reinterpret_cast<void**>(&funcptr);                       \
    tmp = dlsym(handle, symbol);                                     \
    if (tmp == NULL) {                                               \
      NDLL_FAIL("dlsym failed on " + symbol + " - " + dlerror());    \
      goto teardown;                                                 \
    }                                                                \
    *cast = tmp;                                                     \
  } while (0)

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByIndex", nvmlInternalDeviceGetHandleByIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceSetCpuAffinity", nvmlInternalDeviceSetCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlDeviceClearCpuAffinity", nvmlInternalDeviceClearCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);

  symbolsLoaded = 1;
  return NDLLSuccess;

  teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceSetCpuAffinity = NULL;
  nvmlInternalDeviceClearCpuAffinity = NULL;

  if (nvmlhandle != NULL) dlclose(nvmlhandle);
  return NDLLError;
}


NDLLError_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLSuccess;
  }
  nvmlReturn_t ret = nvmlInternalInit();
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlInit() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalShutdown();
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlShutdown() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device);
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlDeviceGetHandleByPciBusId() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlDeviceGetHandleByIndex(const int device_id, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByIndex == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetHandleByIndex(device_id, device);
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlDeviceGetHandleByIndex() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceGetIndex(device, index);
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlDeviceGetIndex() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalDeviceSetCpuAffinity == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceSetCpuAffinity(device);
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlDeviceSetCpuAffinity() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

NDLLError_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalInit == NULL) {
    NDLL_FAIL("lib wrapper not initialized.");
    return NDLLError;
  }
  nvmlReturn_t ret = nvmlInternalDeviceClearCpuAffinity(device);
  if (ret != NVML_SUCCESS) {
    NDLL_FAIL("nvmlDeviceClearCpuAffinity() failed: " +
      nvmlInternalErrorString(ret));
    return NDLLError;
  }
  return NDLLSuccess;
}

}  // namespace nvml

}  // namespace ndll
