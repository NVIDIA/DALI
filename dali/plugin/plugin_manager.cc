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

#include <dlfcn.h>
#include "dali/plugin/plugin_manager.h"
#include "dali/core/error_handling.h"

namespace dali {

void PluginManager::LoadLibrary(const std::string& lib_path, bool global_symbols) {
    // dlopen is thread safe
    int flags = global_symbols ? RTLD_GLOBAL : RTLD_LOCAL;
    flags |= RTLD_LAZY;
    auto handle = dlopen(lib_path.c_str(), flags);
    DALI_ENFORCE(handle != nullptr, "Failed to load library: " + std::string(dlerror()));
}

}  // namespace dali
