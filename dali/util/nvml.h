// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <pthread.h>
#include <sys/sysinfo.h>

#include <mutex>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"
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
 * @brief Gets the CPU affinity mask using NVML,
 *        respecting previously set mask.
 */
inline void GetNVMLAffinityMask(cpu_set_t * mask, size_t num_cpus) {
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  // Get the ideal placement from NVML
  size_t cpu_set_size = (num_cpus + 63) / 64;
  std::vector<unsigned long> nvml_mask_container(cpu_set_size);  // NOLINT(runtime/int)
  auto * nvml_mask = nvml_mask_container.data();
  nvmlDevice_t device;
  DALI_CALL(wrapNvmlDeviceGetHandleByIndex(device_idx, &device));
  DALI_CALL(wrapNvmlDeviceGetCpuAffinity(device, cpu_set_size, nvml_mask));

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
      DALI_WARN("Requested setting affinity to core " + to_string(core) +
                " but only " + to_string(num_cpus) + " cores available. " +
                "Ignoring...");
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
  DALI_CALL(wrapNvmlShutdown());
}

}  // namespace nvml
}  // namespace dali

#define NVML_CALL(code)                                    \
  do {                                                     \
    nvmlReturn_t status = code;                            \
    if (status != NVML_SUCCESS) {                          \
      dali::string error = dali::string("NVML error \"") + \
        nvmlErrorString(status) + "\"";                    \
      DALI_FAIL(error);                                    \
    }                                                      \
  } while (0)

#endif  // DALI_UTIL_NVML_H_
