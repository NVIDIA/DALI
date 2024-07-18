// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/nvml.h"
#include <pthread.h>
#include <sys/sysinfo.h>
#include <vector>

namespace dali {
namespace nvml {
namespace impl {


float GetDriverVersion() {
  if (!nvmlIsInitialized()) {
    return 0;
  }

  float driver_version = 0;
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];

  CUDA_CALL(nvmlSystemGetDriverVersion(version, sizeof version));
  driver_version = std::stof(version);
  return driver_version;
}

int GetCudaDriverVersion() {
  if (!nvmlIsInitialized()) {
    return 0;
  }

  int driver_version = 0;

  CUDA_CALL(nvmlSystemGetCudaDriverVersion(&driver_version));
  return driver_version;
}

}  // namespace impl

namespace {
std::string uuid_str(const char *uuid, const char *prefix = "GPU-") {
  char buf[64] = {};
  char *ptr = buf;
  ptr += snprintf(ptr, sizeof(buf), "%s", prefix);
  for (int i = 0; i < 16; i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10)
      *ptr++ = '-';
    ptr += snprintf(ptr, sizeof(buf) - (ptr - buf), "%02x", static_cast<unsigned char>(uuid[i]));
  }
  *ptr = 0;
  return buf;
}
}  // namespace

nvmlDevice_t nvmlGetDeviceHandleForCUDA(int cuda_idx) {
  nvmlDevice_t dev;
  cudaDeviceProp prop{};
  CUDA_CALL(cudaGetDeviceProperties(&prop, cuda_idx));
  auto uuid = uuid_str(prop.uuid.bytes);
  auto err = nvmlDeviceGetHandleByUUID(uuid.c_str(), &dev);
  if (err == NVML_ERROR_NOT_FOUND) {
    uuid[0] = 'M';
    uuid[1] = 'I';
    uuid[2] = 'G';
    CUDA_CALL(nvmlDeviceGetHandleByUUID(uuid.c_str(), &dev));
  } else {
    CUDA_CALL(err);
  }
  return dev;
}

void GetNVMLAffinityMask(cpu_set_t *mask, size_t num_cpus) {
  if (!nvmlIsInitialized()) {
    return;
  }
  int device_idx;
  CUDA_CALL(cudaGetDevice(&device_idx));

  // Get the ideal placement from NVML
  size_t cpu_set_size = (num_cpus + 63) / 64;
  std::vector<unsigned long> nvml_mask_container(cpu_set_size);  // NOLINT(runtime/int)
  auto * nvml_mask = nvml_mask_container.data();
  nvmlDevice_t device = nvmlGetDeviceHandleForCUDA(device_idx);
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

void SetCPUAffinity(int core) {
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


}  // namespace nvml
}  // namespace dali
