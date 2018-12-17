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

#include <gtest/gtest.h>
#include "dali/plugin/plugin_manager.h"
#include "dali/test/dali_test_utils.h"

const char kNonExistingLibName[] = "not_a_dali_plugin.so";

static const std::string& DummyPluginLibPath() {
    static const std::string plugin_lib = dali::CurrentExecutableDir() + "/libcustomdummyplugin.so";
    return plugin_lib;
}

TEST(PluginManagerTest, LoadLibraryFail) {
    EXPECT_THROW(
        dali::PluginManager::LoadLibrary(kNonExistingLibName),
        std::runtime_error);
}

TEST(PluginManagerTest, LoadLibraryOK) {
    EXPECT_NO_THROW(
        dali::PluginManager::LoadLibrary(DummyPluginLibPath()) );
}

TEST(PluginManagerTest, LoadingSameLibraryTwiceShouldBeOk) {
    for (int i = 0; i < 2; i++) {
        EXPECT_NO_THROW(
            dali::PluginManager::LoadLibrary(DummyPluginLibPath()) );
    }
}
