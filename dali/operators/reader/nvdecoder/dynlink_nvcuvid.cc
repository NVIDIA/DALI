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
#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"

namespace {

static char __DriverLibName[] = "libnvcuvid.so";
static char __DriverLibName1[] = "libnvcuvid.so.1";

using CUVIDDRIVER = void *;

CUVIDDRIVER loadNvcuvidLibrary() {
  CUVIDDRIVER ret = nullptr;

  ret = dlopen(__DriverLibName, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__DriverLibName1, RTLD_NOW);

    if (!ret) {
      printf("dlopen \"%s\" failed!\n", __DriverLibName);
    }
  }
  return ret;
}

}  // namespace

void *NvcuvidLoadSymbol(const char *name) {
  static CUVIDDRIVER nvcuvidDrvLib = loadNvcuvidLibrary();
  void *ret = nvcuvidDrvLib ? dlsym(nvcuvidDrvLib, name) : nullptr;
  return ret;
}

bool cuvidInitChecked() {
  static CUVIDDRIVER nvcuvidDrvLib = loadNvcuvidLibrary();
  return nvcuvidDrvLib != nullptr;
}

bool cuvidIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvcuvidLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
