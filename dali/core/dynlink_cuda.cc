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
#include <mutex>
#include <string>
#include "dali/core/dynlink_cuda.h"

#include <dlfcn.h>

namespace {

typedef void *CUDADRIVER;

static char __CudaLibName[] = "libcuda.so";
static char __CudaLibName1[] = "libcuda.so.1";

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

void *LoadSymbol(const std::string &name) {
  static CUDADRIVER cudaDrvLib = loadCudaLibrary();
  void *ret = cudaDrvLib ? dlsym(cudaDrvLib, name.c_str()) : nullptr;
  if (!ret) {
    printf("Failed to find required function \"%s\" in %s\n", name.c_str(), __CudaLibName);
  }
  return ret;
}

} // namespace

// it is defined in the generated file
typedef void *tLoadSymbol(const std::string &name);
void CudaSetSymbolLoader(tLoadSymbol loader_func);

bool cuInitChecked() {
  static std::mutex m;
  static bool initialized = false;

  if (initialized)
    return true;

  std::lock_guard<std::mutex> lock(m);

  if (initialized)
      return true;

  // set symbol loader for this library
  CudaSetSymbolLoader(LoadSymbol);
  static CUresult res = cuInit(0);
  initialized = (res == CUDA_SUCCESS);
  return initialized;
}
