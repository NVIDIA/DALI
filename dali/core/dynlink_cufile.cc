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
#include "dali/core/dynlink_cufile.h"

namespace {

typedef void* CUFILE;

static const char __CufileLibName[] = "libcufile.so";
static const char __CufileLibName1[] = "libcufile.so.0";

CUFILE loadCufileLibrary() {
  CUFILE ret = nullptr;

  ret = dlopen(__CufileLibName1, RTLD_NOW);

  if (!ret) {
    ret = dlopen(__CufileLibName, RTLD_NOW);

    if (!ret) {
      printf("dlopen \"%s\" failed!\n", __CufileLibName);
    }
  }
  return ret;
}

}  // namespace

void *CufileLoadSymbol(const char *name) {
  static CUFILE cufileDrvLib = loadCufileLibrary();
  void *ret = cufileDrvLib ? dlsym(cufileDrvLib, name) : nullptr;
  return ret;
}
