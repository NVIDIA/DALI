// Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/plugin/plugin_manager.h"
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <utility>
#include "dali/core/error_handling.h"

namespace fs = std::filesystem;

namespace dali {

void PluginManager::LoadLibrary(const std::string& lib_path, bool global_symbols, bool allow_fail) {
  // dlopen is thread safe
  int flags = global_symbols ? RTLD_GLOBAL : RTLD_LOCAL;
  flags |= RTLD_LAZY;
  LOG_LINE << "Loading " << lib_path << "\n";
  auto handle = dlopen(lib_path.c_str(), flags);
  if (handle == nullptr) {
    std::string err_msg =
        std::string("Failed to load library ") + lib_path + ": " + std::string(dlerror());
    if (allow_fail) {
      std::cerr << err_msg << "\n";
    } else {
      DALI_FAIL(err_msg);
    }
  }
}

inline const std::string& DefaultPluginPath() {
  static const std::string path = [&]() -> std::string {
    Dl_info info;
    if (dladdr((const void*)DefaultPluginPath, &info)) {
      fs::path path(info.dli_fname);
      // use the directory of the current shared-object file as starting point to autodiscover the
      // plugin default directory
      // ~/.local/lib/python3.8/site-packages/nvidia/dali/libdali.so ->
      //     ~/.local/lib/python3.8/site-packages/nvidia/dali/plugins/{plugin_name}/libdali_{plugin_name}.so
      path = path.parent_path();
      path /= "plugin";
      return path.string();
    }
    return {};
  }();
  return path;
}

inline void PluginManager::LoadDirectory(const std::string& path, bool global_symbols,
                                         bool allow_fail) {
  std::vector<std::string> plugin_paths;
  if (!fs::is_directory(path)) {
    LOG_LINE << path << " is not a directory. Nothing to load\n";
    return;
  }
  for (const auto& fpath : fs::recursive_directory_iterator(path)) {
    // pos=0 limits the search to the prefix
    if (fpath.path().stem().string().rfind("libdali_", 0) == 0 &&
        fpath.path().extension() == ".so") {
      // filename starts with libdali_ and ends with .so
      auto p = fpath.path().string();
      PluginManager::LoadLibrary(std::move(p), global_symbols, allow_fail);
    }
  }
}

inline void PreloadPluginList(const std::string& dali_preload_plugins) {
  const char delimiter = ':';
  std::string preload(dali_preload_plugins);
  size_t index = 0;
  size_t previous = 0;
  do {
    index = dali_preload_plugins.find(delimiter, previous);
    auto plugin_path = (index != std::string::npos) ?
      dali_preload_plugins.substr(previous, index - previous) :
      dali_preload_plugins.substr(previous);
    if (fs::is_directory(plugin_path)) {
      PluginManager::LoadDirectory(plugin_path, false, true);
    } else {
      PluginManager::LoadLibrary(plugin_path, false, true);
    }
    previous = index + 1;
  } while (index != std::string::npos);
}

void PluginManager::LoadDefaultPlugins() {
  static bool run_once = []() {
    std::string preload_plugins_str = "default";
    const char* dali_preload_plugins = std::getenv("DALI_PRELOAD_PLUGINS");
    if (dali_preload_plugins)
      preload_plugins_str = dali_preload_plugins;
    if (preload_plugins_str == "default") {
      PluginManager::LoadDirectory(DefaultPluginPath(), false, true);
    } else {
      PreloadPluginList(preload_plugins_str);
    }
    return true;
  }();
  (void) run_once;
}

}  // namespace dali
