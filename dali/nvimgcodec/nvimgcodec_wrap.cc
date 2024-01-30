// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dlfcn.h>
#include <nvimgcodec_version.h>
#include <stdio.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

#define STR_IMPL_(x) #x
#define STR(x) STR_IMPL_(x)
#define FULL_VER_STR               \
  STR(NVIMGCODEC_EXT_API_VER_MAJOR) \
  "." STR(NVIMGCODEC_EXT_API_VER_MINOR) "." STR(NVIMGCODEC_EXT_API_VER_PATCH)
#define MAJOR_VER_STR STR(NVIMGCODEC_EXT_API_VER_MAJOR)

namespace {

typedef void *NVIMGCODECDRIVER;

static const char __NvimgcodecLibNameFullVer[] = "libnvimgcodec.so." FULL_VER_STR;
static const char __NvimgcodecLibNameMajorVer[] = "libnvimgcodec.so." MAJOR_VER_STR;
static const char __NvimgcodecLibName[] = "libnvimgcodec.so";
static const char __NvimgcodecLibDefaultPathFullVer[] =
    NVIMGCODEC_DEFAULT_INSTALL_PATH "/lib64/libnvimgcodec.so." FULL_VER_STR;
static const char __NvimgcodecLibDefaultPathMajorVer[] =
    NVIMGCODEC_DEFAULT_INSTALL_PATH "/lib64/libnvimgcodec.so." MAJOR_VER_STR;
static const char __NvimgcodecLibDefaultPath[] =
    NVIMGCODEC_DEFAULT_INSTALL_PATH "/lib64/libnvimgcodec.so";

NVIMGCODECDRIVER loadNvimgcodecLibrary() {
  static const char *paths[] = {__NvimgcodecLibNameFullVer,
                                __NvimgcodecLibNameMajorVer,
                                __NvimgcodecLibName,
                                __NvimgcodecLibDefaultPathFullVer,
                                __NvimgcodecLibDefaultPathMajorVer,
                                __NvimgcodecLibDefaultPath};
  NVIMGCODECDRIVER ret = nullptr;
  for (const char *path : paths) {
    ret = dlopen(path, RTLD_NOW);
    if (ret)
      break;
  }
  if (!ret)
    throw std::runtime_error(
        "dlopen libnvimgcodec.so failed!. Please install nvimagecodec: See "
        "https://developer.nvidia.com/nvimgcodec-downloads or simply do `pip install "
        "nvidia-nvimgcodec-cu${CUDA_MAJOR_VERSION}`.");
  return ret;
}

}  // namespace

void *NvimgcodecLoadSymbol(const char *name) {
  static NVIMGCODECDRIVER nvimgcodecDrvLib = loadNvimgcodecLibrary();
  void *ret = nvimgcodecDrvLib ? dlsym(nvimgcodecDrvLib, name) : nullptr;
  return ret;
}

bool nvimgcodecIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void *> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvimgcodecLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
