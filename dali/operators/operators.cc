// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators.h"
#include "dali/core/api_helper.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/npp/npp.h"
#include "dali/plugin/plugin_manager.h"

#if DALI_USE_NVJPEG
#include "dali/operators/decoder/nvjpeg/nvjpeg_helper.h"
#endif

#include <dlfcn.h>
#include <nvimgcodec.h>
#include <filesystem>

namespace fs = std::filesystem;

/*
 * The point of these functions is to force the linker to link against dali_operators lib
 * and not optimize-out symbols from dali_operators
 *
 * The functions to reference, when one needs to make sure DALI operators
 * shared object is actually linked against.
 */

namespace dali {

inline void loadPlugin(const std::string& path) {
  std::cout << "Loading " << path << std::endl;
  PluginManager::LoadLibrary(path);
}

inline std::string GetDefaultPluginPath()
{
    Dl_info info;
    if (dladdr((const void*)GetDefaultPluginPath, &info)) {
      fs::path path(info.dli_fname);
      // use the directory of the current shared-object file as starting point to autodiscover the plugins
      // ~/.local/lib/python3.8/site-packages/nvidia/dali/libdali_operators.so ->
      //     ~/.local/lib/python3.8/site-packages/nvidia/dali/plugins/{plugin_name}/libdali_{plugin_name}.so
      path = path.parent_path();
      path /= "plugin";
      return path.string();
    }
    DALI_FAIL("Can't find the default plugin path");
    return "";
}

inline void AutodiscoverPluginsLibs() {
  const char *dali_autodiscover_plugins = std::getenv("DALI_AUTODISCOVER_PLUGINS");
  int autodiscover_plugins = dali_autodiscover_plugins ? atoi(dali_autodiscover_plugins) : 0;
  if (!autodiscover_plugins)
    return;

  std::cout << "Auto discovering DALI plugins\n";
  auto path = GetDefaultPluginPath();
  std::vector<std::string> plugin_paths;
  if (!fs::is_directory(path)) {
    LOG_LINE << path << " is not a directory. Nothing to load\n";
    return;
  }

  for (const auto& fpath : fs::recursive_directory_iterator(path)) {
    // pos=0 limits the search to the prefix
    if (fpath.path().stem().string().rfind("libdali_", 0) == 0 && fpath.path().extension() == ".so") {
      // filename starts with libdali_ and ends with .so
      loadPlugin(fpath.path().string());
    }
  }
  std::cout << "Auto discovering DALI plugins done\n";
}

inline void PreloadPluginsLibs() {
  const char *dali_preload_plugins_env = std::getenv("DALI_PRELOAD_PLUGINS");
  if (!dali_preload_plugins_env)
    return;
  std::string dali_preload_plugins(dali_preload_plugins_env);

  std::cout << "Preloading DALI plugins\n";
  const char delimiter = ':';
  size_t previous = 0;
  std::string preload(dali_preload_plugins);
  size_t index = dali_preload_plugins.find(delimiter);
  std::vector<std::string> plugins;
  while(index != string::npos) {
    auto plugin_path = dali_preload_plugins.substr(previous, index - previous);
    loadPlugin(plugin_path);
    previous = index + 1;
    index = dali_preload_plugins.find(delimiter, previous);
  }
  auto plugin_path = dali_preload_plugins.substr(previous);
  loadPlugin(plugin_path);
  plugins.push_back(dali_preload_plugins.substr(previous));

  std::cout << "Preloading DALI plugins done\n";
}

DLL_PUBLIC void InitOperatorsLib() {
  (void)CUDAStreamPool::instance();
  PreloadPluginsLibs();
  AutodiscoverPluginsLibs();
}


DLL_PUBLIC int GetNppVersion() {
  return NPPGetVersion();
}

DLL_PUBLIC int GetNvjpegVersion() {
#if DALI_USE_NVJPEG
  return nvjpegGetVersion();
#else
  return -1;
#endif
}

DLL_PUBLIC int GetNvimgcodecVersion() {
#if not(NVIMAGECODEC_ENABLED)
  return -1;
#else
  nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES,
                                    sizeof(nvimgcodecProperties_t), 0};
  if (NVIMGCODEC_STATUS_SUCCESS != nvimgcodecGetProperties(&properties))
    return -1;
  return static_cast<int>(properties.version);
#endif
}

DLL_PUBLIC void EnforceMinimumNvimgcodecVersion() {
#if NVIMAGECODEC_ENABLED
  auto version = GetNvimgcodecVersion();
  if (version == -1) {
    throw std::runtime_error("Failed to check the version of nvimgcodec.");
  }
  int major = NVIMGCODEC_MAJOR_FROM_SEMVER(version);
  int minor = NVIMGCODEC_MINOR_FROM_SEMVER(version);
  int patch = NVIMGCODEC_PATCH_FROM_SEMVER(version);
  if (major < NVIMGCODEC_VER_MAJOR || minor < NVIMGCODEC_VER_MINOR ||
      patch < NVIMGCODEC_VER_PATCH) {
    std::stringstream ss;
    ss << "DALI requires nvImageCodec at minimum version" << NVIMGCODEC_VER_MAJOR << "."
       << NVIMGCODEC_VER_MINOR << "." << NVIMGCODEC_VER_PATCH << ", but got " << major << "."
       << minor << "." << patch
       << ". Please upgrade: See https://developer.nvidia.com/nvimgcodec-downloads or simply do "
          "`pip install nvidia-nvimgcodec-cu${CUDA_MAJOR_VERSION} --upgrade`.";
    throw std::runtime_error(ss.str());
  }
#endif
}

}  // namespace dali

extern "C" DLL_PUBLIC void daliInitOperators() {}
