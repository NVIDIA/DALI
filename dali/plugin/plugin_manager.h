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

#ifndef DALI_PLUGIN_PLUGIN_MANAGER_H_
#define DALI_PLUGIN_PLUGIN_MANAGER_H_

#include <string>
#include <vector>
#include "dali/core/common.h"

namespace dali {

class DLL_PUBLIC PluginManager {
 public:
  /**
   * @brief Load plugin library
   * @remarks Will invoke dlopen()
   * @param [in] lib_path path to the plugin library, e.g. "/usr/lib/libcustomplugin.so"
   * @param [in] global_symbols if true, the library is loaded with RTLD_GLOBAL flag or equivalent
   *                            otherwise, RTLD_LOCAL is used
   * @param [in] allow_fail if true, not being able to load a library won't result in a hard error
   * @throws std::runtime_error if the library could not be loaded
   */
  static DLL_PUBLIC void LoadLibrary(const std::string& lib_path, bool global_symbols = false,
                                     bool allow_fail = false);

  /**
   * @brief Load plugin directory. The plugin paths will have the following pattern:
   *        {lib_path}/{subpath}/libdali_{plugin_name}.so
   * @param [in] lib_path path to the root directory where the plugins are located
   * @param [in] global_symbols if true, the library is loaded with RTLD_GLOBAL flag or equivalent
   *                            otherwise, RTLD_LOCAL is used
   * @param [in] allow_fail if true, not being able to load a library won't result in a hard error
   * @throws std::runtime_error if the library could not be loaded
   */
  static DLL_PUBLIC void LoadDirectory(const std::string& lib_path, bool global_symbols = false,
                                       bool allow_fail = false);

  /**
   * @brief Load default plugin library
   * @remarks DALI_PRELOAD_PLUGINS are environment variables that can be used to control what
   * plugins are loaded. If the variable is set, it is interpreted as a list of paths separated
   * by colon (:), where each element can be a directory or library path.
   * If not set, the "default" path is scanned, which is a subdirectory called plugin under the
   * directory where the DALI library is installed.
   */
  static DLL_PUBLIC void LoadDefaultPlugins();
};

}  // namespace dali

#endif  // DALI_PLUGIN_PLUGIN_MANAGER_H_
