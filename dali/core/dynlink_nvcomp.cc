// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Set by the build system to match the nvCOMP version DALI was built against.
// There's no fallback to the unversioned libnvcomp.so, which may be any (ABI-incompatible) version.
static const char __nvCompfileLibName[] = DALI_NVCOMP_SONAME;

NVCOMP loadNvCompFileLibrary() {
  NVCOMP ret = dlopen(__nvCompfileLibName, RTLD_NOW);

  if (!ret) {
#if FOR_CONDA_ENABLED
    fprintf(stderr, "dlopen %s failed with error %s!. Please install libnvcomp: "
            "`conda install -c conda-forge libnvcomp`.\n", __nvCompfileLibName, dlerror());
#else
    fprintf(stderr, "dlopen %s failed with error %s!. Please install libnvcomp.\n",
            __nvCompfileLibName, dlerror());
#endif
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
