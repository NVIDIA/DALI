// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/plugin/plugin_manager.h"
#include "dali/error_handling.h"

namespace dali {

PluginManager& PluginManager::Instance() {
    static PluginManager instance;
    return instance;
}

PluginManager::~PluginManager() {
    // WARNING! Calling dlclose will produce a crash because the lifecycle
    //          of SchemaRegistry and OperatorRegistry is not managed
    // TODO(janton): consider doing dlclose after refactoring of the registry's lifecycle
#if 0
    for (auto handle : handles_) {
        dlclose(handle);
    }
#endif
}

void PluginManager::LoadLibrary(const std::string& lib_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    DALI_ENFORCE(handle != nullptr, "Failed to load library: " + std::string(dlerror()));
    handles_.push_back(handle);
}

}  // namespace dali
