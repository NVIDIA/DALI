// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime_api.h>
#include <mutex>
#include <string>
#include "dali/util/nvml_wrap.h"
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

namespace dali {

class NvmlError : public std::runtime_error {
 public:
  explicit NvmlError(nvmlReturn_t result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(nvmlReturn_t result) {
    switch (result) {
      case NVML_SUCCESS:
        return "nvml operation was successful";
      case NVML_ERROR_UNINITIALIZED:
        return "nvml was not first initialized with nvmlInit()";
      case NVML_ERROR_INVALID_ARGUMENT:
        return "a nvml supplied argument is invalid";
      case NVML_ERROR_NOT_SUPPORTED:
        return "The nvml requested operation is not available on target device";
      case NVML_ERROR_NO_PERMISSION:
        return "The nvml current user does not have permission for operation";
      case NVML_ERROR_ALREADY_INITIALIZED:
        return "Deprecated: Multiple initializations are now allowed through ref counting";
      case NVML_ERROR_NOT_FOUND:
        return "A nvml query to find an object was unsuccessful";
      case NVML_ERROR_INSUFFICIENT_SIZE:
        return "A nvml input argument is not large enough";
      case NVML_ERROR_INSUFFICIENT_POWER:
        return "A nvml device's external power cables are not properly attached";
      case NVML_ERROR_DRIVER_NOT_LOADED:
        return "nvml: NVIDIA driver is not loaded";
      case NVML_ERROR_TIMEOUT:
        return "nvml user provided timeout passed";
      case NVML_ERROR_IRQ_ISSUE:
        return "nvml: NVIDIA Kernel detected an interrupt issue with a GPU";
      case NVML_ERROR_LIBRARY_NOT_FOUND:
        return "NVML Shared Library couldn't be found or loaded";
      case NVML_ERROR_FUNCTION_NOT_FOUND:
        return "Local version of NVML doesn't implement this function";
      case NVML_ERROR_CORRUPTED_INFOROM:
        return "nvml: infoROM is corrupted";
      case NVML_ERROR_GPU_IS_LOST:
        return "nvml: the GPU has fallen off the bus or has otherwise become inaccessible";
      case NVML_ERROR_RESET_REQUIRED:
        return "nvml: the GPU requires a reset before it can be used again";
      case NVML_ERROR_OPERATING_SYSTEM:
        return "nvml: the GPU control device has been blocked by the operating system/cgroups";
      case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        return "nvml: RM detects a driver/library version mismatch";
      case NVML_ERROR_IN_USE:
        return "A nvml operation cannot be performed because the GPU is currently in use";
      case NVML_ERROR_MEMORY:
        return "Nvml insufficient memory";
      case NVML_ERROR_NO_DATA:
        return "Nvml: no data";
#if (CUDART_VERSION >= 11000)
      case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
        return "The nvml requested vgpu operation is not available on target device, becasue ECC is"
               "enabled";
      case NVML_ERROR_INSUFFICIENT_RESOURCES:
        return "Nvml: ran out of critical resources, other than memory";
#endif
      case NVML_ERROR_UNKNOWN:
        return "A nvml internal driver error occurred";
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(nvmlReturn_t result, const char *details) {
    if (details && *details) {
      return make_string("nvml error (", result, "): ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("nvml error (", result, "): ", ErrorString(result));
    }
  }


  nvmlReturn_t result() const { return result_; }

 private:
  nvmlReturn_t result_;
};

class NvmlBadAlloc : public CUDABadAlloc {};

template <>
inline void cudaResultCheck<nvmlReturn_t>(nvmlReturn_t status) {
  switch (status) {
  case NVML_SUCCESS:
    return;
  case NVML_ERROR_MEMORY:
    throw dali::NvmlBadAlloc();
  default:
    throw dali::NvmlError(status);
  }
}

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
  CUDA_CALL(nvmlInitChecked());
}

/**
 * @brief Gets the CPU affinity mask using NVML,
 *        respecting previously set mask.
 */
void GetNVMLAffinityMask(cpu_set_t * mask, size_t num_cpus);

/**
 * @brief Sets the CPU affinity for the calling thread
 */
void SetCPUAffinity(int core = -1);

inline void Shutdown() {
  std::lock_guard<std::mutex> lock(Mutex());
  if (!nvmlIsInitialized()) {
    return;
  }
  CUDA_CALL(nvmlShutdown());
}


class NvmlInstance {
 public:
  static NvmlInstance CreateNvmlInstance() {
    return NvmlInstance(true);
  }

  explicit NvmlInstance(bool init = false) {
    if (init) {
      Init();
      is_created_ = true;
    }
  }

  NvmlInstance(const NvmlInstance &) = delete;

  NvmlInstance &operator=(const NvmlInstance &) = delete;

  inline NvmlInstance(NvmlInstance &&other) : is_created_(other.is_created_) {
    other.is_created_ = false;
  }

  inline NvmlInstance &operator=(NvmlInstance &&other) {
    std::swap(is_created_, other.is_created_);
    other.~NvmlInstance();
    return *this;
  }

  ~NvmlInstance() {
    if (is_created_) {
      Shutdown();
      is_created_ = false;
    }
  }

 private:
  bool is_created_ = false;
};

/**
 * Checks, whether CUDA11-proper NVML functions have been successfully loaded
 */
inline bool HasCuda11NvmlFunctions() {
  if (!nvmlIsInitialized()) {
    return false;
  }
  return nvmlHasCuda11NvmlFunctions();
}


namespace impl {

float GetDriverVersion();
int GetCudaDriverVersion();

}  // namespace impl


inline float GetDriverVersion() {
  static float version = impl::GetDriverVersion();
  return version;
}


inline int GetCudaDriverVersion() {
  static int version = impl::GetCudaDriverVersion();
  return version;
}


#if (CUDART_VERSION >= 11000)

/**
 * Agregates info about the device
 */
struct DeviceProperties {
  nvmlBrandType_t type;
  int cap_major;
  int cap_minor;
};

/**
 * Obtains info about device with given ID
 *
 * @throws std::runtime_error
 */

inline DeviceProperties GetDeviceInfo(int device_idx) {
  DeviceProperties ret;
  nvmlDevice_t device;
  CUDA_CALL(nvmlDeviceGetHandleByIndex_v2(device_idx, &device));
  CUDA_CALL(nvmlDeviceGetBrand(device, &ret.type));
  CUDA_CALL(nvmlDeviceGetCudaComputeCapability(device, &ret.cap_major, &ret.cap_minor));
  return ret;
}

/**
 * Checks, if hardware decoder is available for the provided device
 *
 * @throws std::runtime_error
 */
inline bool HasHwDecoder(int device_idx) {
  if (!nvmlIsInitialized()) {
    return false;
  }
  auto info = GetDeviceInfo(device_idx);
  return info.type == NVML_BRAND_TESLA &&
         (info.cap_major == 8 || info.cap_major == 9) &&
         info.cap_minor == 0;
}

/**
 * Checks, if hardware decoder is available in all possible devices
 *
 * @throws std::runtime_error
 */
inline bool HasHwDecoder() {
  if (!nvmlIsInitialized()) {
    return false;
  }
  unsigned int device_count;
  CUDA_CALL(nvmlDeviceGetCount_v2(&device_count));
  for (unsigned int device_idx = 0; device_idx < device_count; device_idx++) {
    if (HasHwDecoder(device_idx)) return true;
  }
  return false;
}

inline bool isHWDecoderSupported() {
  if (nvml::HasCuda11NvmlFunctions()) {
    return nvml::HasHwDecoder();
  }
  return false;
}
#else

inline bool isHWDecoderSupported()  {
  return false;
}

#endif

}  // namespace nvml
}  // namespace dali

#endif  // DALI_UTIL_NVML_H_
