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

#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <stdexcept>

#include "dali/npp/npp.h"

namespace {

typedef void* NPPCDRIVER;
typedef void* NPPICCDRIVER;
typedef void* NPPIGDRIVER;

static const char __NppcLibName[] = "libnppc.so";
static const char __NppiccLibName[] = "libnppicc.so";
static const char __NppigLibName[] = "libnppig.so";
#if CUDA_VERSION >= 12000
static const char __NppcLibNameCuVer[] = "libnppc.so.12";
static const char __NppiccLibNameCuVer[] = "libnppicc.so.12";
static const char __NppigLibNameCuVer[] = "libnppig.so.12";
#elif CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
static const char __NppcLibNameCuVer[] = "libnppc.so.11";
static const char __NppiccLibNameCuVer[] = "libnppicc.so.11";
static const char __NppigLibNameCuVer[] = "libnppig.so.11";
#else
static const char __NppcLibNameCuVer[] = "libnppc.so.10";
static const char __NppiccLibNameCuVer[] = "libnppicc.so.10";
static const char __NppigLibNameCuVer[] = "libnppig.so.10";
#endif

NPPCDRIVER loadNppcLibrary() {
  NPPCDRIVER ret = nullptr;

  ret = dlopen(__NppcLibNameCuVer, RTLD_NOW);
  if (!ret) {
    ret = dlopen(__NppcLibName, RTLD_NOW);
    if (!ret) {
      throw std::runtime_error("dlopen libnppc.so failed!. Please install "
          "CUDA toolkit or NPP python wheel.");
    }
  }
  return ret;
}

NPPICCDRIVER loadNppiccLibrary() {
  NPPICCDRIVER ret = nullptr;

  ret = dlopen(__NppiccLibNameCuVer, RTLD_NOW);
  if (!ret) {
    ret = dlopen(__NppiccLibName, RTLD_NOW);
    if (!ret) {
      throw std::runtime_error("dlopen libnppicc.so failed!. Please install "
          "CUDA toolkit or NPP python wheel.");
    }
  }
  return ret;
}

NPPIGDRIVER loadNppigLibrary() {
  NPPIGDRIVER ret = nullptr;

  ret = dlopen(__NppigLibNameCuVer, RTLD_NOW);
  if (!ret) {
    ret = dlopen(__NppigLibName, RTLD_NOW);
    if (!ret) {
      throw std::runtime_error("dlopen libnppig.so failed!. Please install "
          "CUDA toolkit or NPP python wheel.");
    }
  }
  return ret;
}

}  // namespace

// Load a symbol from all NPP libs that we use, to provide a unified interface
void *NppLoadSymbol(const char *name) {
  static NPPCDRIVER nppcDrvLib = loadNppcLibrary();
  static NPPICCDRIVER nppiccDrvLib = loadNppiccLibrary();
  static NPPIGDRIVER nppigDrvLib = loadNppigLibrary();
  // check processing library, core later if symbol not found
  void *ret = nppiccDrvLib ? dlsym(nppiccDrvLib, name) : nullptr;
  if (!ret) {
    ret = nppigDrvLib ? dlsym(nppigDrvLib, name) : nullptr;
    if (!ret) {
      ret = nppcDrvLib ? dlsym(nppcDrvLib, name) : nullptr;
    }
  }
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
