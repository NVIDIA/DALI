// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>

#include "dali/util/nvml_wrap.h"


namespace {

typedef void* NVMLRIVER;

static const char __NvmlLibName[] = "libnvidia-ml.so";
static const char __NvmlLibName1[] = "libnvidia-ml.so.1";

NVMLRIVER loadNvmlLibrary() {
  NVMLRIVER ret = nullptr;

  ret = dlopen(__NvmlLibName1, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__NvmlLibName, RTLD_NOW);

    if (!ret) {
      printf("dlopen \"%s\" failed!\n", __NvmlLibName);
    }
  }
  return ret;
}

std::atomic_bool initialized{false};

}  // namespace

void *NvmlLoadSymbol(const char *name) {
  static NVMLRIVER nvmlDrvLib = loadNvmlLibrary();
  void *ret = nvmlDrvLib ? dlsym(nvmlDrvLib, name) : nullptr;
  return ret;
}

nvmlReturn_t nvmlInitChecked() {
  nvmlReturn_t ret = nvmlInit();
  if (ret != NVML_SUCCESS) {
    DALI_WARN("nvmlInitChecked failed: ", nvmlErrorString(ret));
  }
  initialized = true;
  return ret;
}

bool nvmlIsInitialized(void) {
  return initialized;
}

bool nvmlIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvmlLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}

bool nvmlHasCuda11NvmlFunctions(void) {
  return nvmlIsSymbolAvailable("nvmlDeviceGetCount_v2") &&
         nvmlIsSymbolAvailable("nvmlDeviceGetHandleByIndex_v2") &&
         nvmlIsSymbolAvailable("nvmlDeviceGetCudaComputeCapability") &&
         nvmlIsSymbolAvailable("nvmlDeviceGetBrand") &&
         nvmlIsSymbolAvailable("nvmlDeviceGetCpuAffinityWithinScope");
}
