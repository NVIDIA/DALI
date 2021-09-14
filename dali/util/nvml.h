// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pthread.h>
#include <sys/sysinfo.h>
#include <mutex>
#include <vector>
#include <string>
#include "dali/core/cuda_utils.h"
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
      return make_string("nvml error: ", result, " ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("nvml error: ", result, " ", ErrorString(result));
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
inline void GetNVMLAffinityMask(cpu_set_t * mask, size_t num_cpus) {
  if (!nvmlIsInitialized()) {
    return;
  }
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  // Get the ideal placement from NVML
  size_t cpu_set_size = (num_cpus + 63) / 64;
  std::vector<unsigned long> nvml_mask_container(cpu_set_size);  // NOLINT(runtime/int)
  auto * nvml_mask = nvml_mask_container.data();
  nvmlDevice_t device;
  CUDA_CALL(nvmlDeviceGetHandleByIndex(device_idx, &device));
  #if (CUDART_VERSION >= 11000)
    if (nvmlHasCuda11NvmlFunctions()) {
      CUDA_CALL(nvmlDeviceGetCpuAffinityWithinScope(device, cpu_set_size, nvml_mask,
                                                        NVML_AFFINITY_SCOPE_SOCKET));
    } else {
      CUDA_CALL(nvmlDeviceGetCpuAffinity(device, cpu_set_size, nvml_mask));
    }
  #else
    CUDA_CALL(nvmlDeviceGetCpuAffinity(device, cpu_set_size, nvml_mask));
  #endif

  // Convert it to cpu_set_t
  cpu_set_t nvml_set;
  CPU_ZERO(&nvml_set);
  const size_t n_bits = sizeof(unsigned long) * 8;  // NOLINT(runtime/int)
  for (size_t i = 0; i < num_cpus; ++i) {
    const size_t position = i % n_bits;
    const size_t index = i / n_bits;
    const unsigned long current_mask = 1ul << position;  // NOLINT(runtime/int)
    const bool cpu_is_set = (nvml_mask[index] & current_mask) != 0;
    if (cpu_is_set) {
        CPU_SET(i, &nvml_set);
    }
  }

  // Get the current affinity mask
  cpu_set_t current_set;
  CPU_ZERO(&current_set);
  pthread_getaffinity_np(pthread_self(), sizeof(current_set), &current_set);

  // AND masks
  CPU_AND(mask, &nvml_set, &current_set);
}

/**
 * @brief Sets the CPU affinity for the calling thread
 */
inline void SetCPUAffinity(int core = -1) {
  std::lock_guard<std::mutex> lock(Mutex());

  size_t num_cpus = get_nprocs_conf();

  cpu_set_t requested_set;
  CPU_ZERO(&requested_set);
  if (core != -1) {
    if (core < 0 || (size_t)core >= num_cpus) {
      DALI_WARN(make_string("Requested setting affinity to core ", core,
                            " but only ", num_cpus, " cores available. Ignoring..."));
      GetNVMLAffinityMask(&requested_set, num_cpus);
    } else {
      CPU_SET(core, &requested_set);
    }
  } else {
    GetNVMLAffinityMask(&requested_set, num_cpus);
  }

  // Set the affinity
  bool at_least_one_cpu_set = false;
  for (std::size_t i = 0; i < num_cpus; i++) {
    at_least_one_cpu_set |= CPU_ISSET(i, &requested_set);
  }
  if (!at_least_one_cpu_set) {
    DALI_WARN("CPU affinity requested by user or recommended by nvml setting"
              " does not meet allowed affinity for given DALI thread."
              " Use taskset tool to check allowed affinity");
    return;
  }

  int error = pthread_setaffinity_np(pthread_self(), sizeof(requested_set), &requested_set);
  if (error != 0) {
    DALI_WARN("Setting affinity failed! Error code: " + to_string(error));
  }
}


inline void Shutdown() {
  std::lock_guard<std::mutex> lock(Mutex());
  if (!nvmlIsInitialized()) {
    return;
  }
  CUDA_CALL(nvmlShutdown());
}

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

}  // namespace impl


inline float GetDriverVersion() {
  static float version = impl::GetDriverVersion();
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
  const int kAmpereComputeCapabilityMajor = 8;
  const int kAmpereComputeCapabilityMinor = 0;
  return info.type == NVML_BRAND_TESLA &&
         info.cap_major == kAmpereComputeCapabilityMajor &&
         info.cap_minor == kAmpereComputeCapabilityMinor;
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
