// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace {

typedef void* NVJPEGDRIVER;

static const char __NvjpegLibName[] = "libnvjpeg.so";
#if CUDA_VERSION >= 12000
static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.12";
#elif CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.11";
#else
static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.10";
#endif

NVJPEGDRIVER loadNvjpegLibrary() {
  NVJPEGDRIVER ret = nullptr;

  ret = dlopen(__NvjpegLibNameCuVer, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__NvjpegLibName, RTLD_NOW);
    if (!ret) {
      throw std::runtime_error("dlopen libnvjpeg.so failed!. Please install "
                                "CUDA toolkit or nvJPEG python wheel.");
    }
  }
  return ret;
}

}  // namespace

void *NvjpegLoadSymbol(const char *name) {
  static NVJPEGDRIVER nvjpegDrvLib = loadNvjpegLibrary();
  void *ret = nvjpegDrvLib ? dlsym(nvjpegDrvLib, name) : nullptr;
  return ret;
}

bool nvjpegIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvjpegLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
