// Copyright (c) 2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include "dali/core/error_handling.h"

namespace fs = std::filesystem;

namespace dali {

void PluginManager::LoadLibrary(const std::string& lib_path, bool global_symbols) {
  // dlopen is thread safe
  int flags = global_symbols ? RTLD_GLOBAL : RTLD_LOCAL;
  flags |= RTLD_LAZY;
  std::cout << "Loading " << lib_path << "\n";
  auto handle = dlopen(lib_path.c_str(), flags);
  DALI_ENFORCE(handle != nullptr, "Failed to load library: " + std::string(dlerror()));
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

inline void PluginManager::LoadDirectory(const std::string& path, bool global_symbols) {
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
      PluginManager::LoadLibrary(std::move(p), global_symbols);
    }
  }
}

inline void PreloadPluginList(const std::string& dali_preload_plugins) {
  const char delimiter = ':';
  size_t previous = 0;
  std::string preload(dali_preload_plugins);
  size_t index = dali_preload_plugins.find(delimiter);
  std::vector<std::string> plugins;
  while (index != string::npos) {
    auto plugin_path = dali_preload_plugins.substr(previous, index - previous);
    if (fs::is_directory(plugin_path)) {
      PluginManager::LoadDirectory(plugin_path);
    } else {
      PluginManager::LoadLibrary(plugin_path);
    }
    previous = index + 1;
    index = dali_preload_plugins.find(delimiter, previous);
  }
  auto plugin_path = dali_preload_plugins.substr(previous);
  if (fs::is_directory(plugin_path)) {
    PluginManager::LoadDirectory(plugin_path);
  } else {
    PluginManager::LoadLibrary(plugin_path);
  }
  plugins.push_back(dali_preload_plugins.substr(previous));
}

void PluginManager::LoadDefaultPlugins() {
  static bool run_once = []() {
    std::string preload_plugins_str = "default";
    const char* dali_preload_plugins = std::getenv("DALI_PRELOAD_PLUGINS");
    if (dali_preload_plugins)
      preload_plugins_str = dali_preload_plugins;
    if (preload_plugins_str == "default") {
      PluginManager::LoadDirectory(DefaultPluginPath());
    } else {
      PreloadPluginList(preload_plugins_str);
    }
    return true;
  }();
  (void) run_once;
}

}  // namespace dali
