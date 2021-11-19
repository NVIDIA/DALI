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

#include "dali/util/npp_wrap.h"


namespace {

typedef void* NPPRIVER;

static const char __NppLibName[] = "libnppc.so";
static const char __NppLibName10[] = "libnppc.so.10";
static const char __NppLibName11[] = "libnppc.so.11";

NPPRIVER loadNppLibrary() {
  NPPRIVER ret = nullptr;

  ret = dlopen(__NppLibName, RTLD_NOW);
  if (!ret) {
    ret = dlopen(__NppLibName10, RTLD_NOW);

    if (!ret) {
      ret = dlopen(__NppLibName11, RTLD_NOW);

      if (!ret) {
        printf("dlopen \"%s\" failed!\n", __NppLibName);
      }
    }
  }
  return ret;
}

}  // namespace

void *NppLoadSymbol(const char *name) {
  static NPPRIVER nppDrvLib = loadNppLibrary();
  void *ret = nppDrvLib ? dlsym(nppDrvLib, name) : nullptr;
  return ret;
}

bool nppIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NppLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
