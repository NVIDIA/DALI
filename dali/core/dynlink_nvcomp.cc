// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dynlink_nvcomp.h"

namespace {

typedef void* NVCOMP;

static const char __nvCompfileLibName[] = "libnvcomp.so";
static const char __nvCompfileLibName1[] = "libnvcomp.so.5";

NVCOMP loadNvCompFileLibrary() {
  NVCOMP ret = nullptr;

  ret = dlopen(__nvCompfileLibName1, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__nvCompfileLibName, RTLD_NOW);

    if (!ret) {
      fprintf(stderr, "dlopen libnvcomp.so failed with error %s!. Please install libnvcomp.\n",
              dlerror());
    }
  }
  return ret;
}

}  // namespace

void *nvCompLoadSymbol(const char *name) {
  static NVCOMP nvCompDrvLib = loadNvCompFileLibrary();
  void *ret = nvCompDrvLib ? dlsym(nvCompDrvLib, name) : nullptr;
  return ret;
}

bool nvCompIsSymbolAvailable(const char *symbol) {
  return nvCompLoadSymbol(symbol) != nullptr;
}
