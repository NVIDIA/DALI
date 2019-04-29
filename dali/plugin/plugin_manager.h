// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/common.h"

namespace dali {

class DLL_PUBLIC PluginManager {
 public:
    /**
     * @brief Load plugin library
     * @remarks Will invoke dlopen()
     * @param [in] lib_path path to the plugin library, e.g. "/usr/lib/libcustomplugin.so"
     * @throws std::runtime_error if the library could not be loaded
     */
    static DLL_PUBLIC void LoadLibrary(const std::string& lib_path);
};

}  // namespace dali

#endif  // DALI_PLUGIN_PLUGIN_MANAGER_H_
