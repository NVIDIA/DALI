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

#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <stdexcept>

#include "dali/kernels/signal/fft/cufft_helper.h"

namespace {

typedef void* CUFFTDIVER;

static const char __CufftLibName[] = "libcufft.so";
#if CUDA_VERSION >= 12000
static const char __CufftLibNameCuVer[] = "libcufft.so.11";
#else
// in cuda 10.x and 11.x it is consistently named libcufft.so.10
static const char __CufftLibNameCuVer[] = "libcufft.so.10";
#endif

CUFFTDIVER loadCufftLibrary() {
  CUFFTDIVER ret = nullptr;

  ret = dlopen(__CufftLibNameCuVer, RTLD_NOW);
  if (!ret) {
    ret = dlopen(__CufftLibName, RTLD_NOW);
    if (!ret) {
      throw std::runtime_error("dlopen libcufft.so failed!. Please install "
                                "CUDA toolkit or cuFFT python wheel.");
    }
  }
  return ret;
}

}  // namespace

void *CufftLoadSymbol(const char *name) {
  static CUFFTDIVER cufftDrvLib = loadCufftLibrary();
  void *ret = cufftDrvLib ? dlsym(cufftDrvLib, name) : nullptr;
  return ret;
}

bool cufftIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = CufftLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
