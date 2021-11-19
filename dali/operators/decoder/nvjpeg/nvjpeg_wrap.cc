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

#include "dali/operators/decoder/nvjpeg/nvjpeg_wrap.h"


namespace {

typedef void* NVJPEGRIVER;

static const char __NvjpegLibName[] = "libnvjpeg.so";

NVJPEGRIVER loadNvjpegLibrary() {
  NVJPEGRIVER ret = nullptr;

  ret = dlopen(__NvjpegLibName, RTLD_NOW);

  if (!ret) {
    printf("dlopen \"%s\" failed!\n", __NvjpegLibName);
  }
  return ret;
}

}  // namespace

void *NvjpegLoadSymbol(const char *name) {
  static NVJPEGRIVER nvjpegDrvLib = loadNvjpegLibrary();
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
