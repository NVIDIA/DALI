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

#include <dlfcn.h>
#include <stdio.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "dali/core/dynlink_cuda.h"

namespace {

typedef void* CUDADRIVER;

static const char __CudaLibName[] = "libcuda.so";
static const char __CudaLibName1[] = "libcuda.so.1";

CUDADRIVER loadCudaLibrary() {
  CUDADRIVER ret = nullptr;

  ret = dlopen(__CudaLibName1, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__CudaLibName, RTLD_NOW);

    if (!ret) {
      printf("dlopen \"%s\" failed!\n", __CudaLibName);
    }
  }
  return ret;
}

}  // namespace

void *CudaLoadSymbol(const char *name) {
  static CUDADRIVER cudaDrvLib = loadCudaLibrary();
  void *ret = cudaDrvLib ? dlsym(cudaDrvLib, name) : nullptr;
  return ret;
}

bool cuInitChecked() {
  static CUresult res = cuInit(0);
  return res == CUDA_SUCCESS;
}

bool cuIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = CudaLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
