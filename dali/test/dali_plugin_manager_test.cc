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

#include <gtest/gtest.h>
#include "dali/plugin/plugin_manager.h"
#include "dali/test/dali_test_matching.h"
#include "dali/test/dali_test_utils.h"

static const std::string& DummyPluginLibPath() {
    static const std::string plugin_lib = dali::CurrentExecutableDir() + "/libcustomdummyplugin.so";
    return plugin_lib;
}

namespace other_ns {

template <typename ImgType>
class DummyTest : public ::dali::GenericMatchingTest<ImgType> {
};

typedef ::testing::Types<::dali::RGB> Types;
TYPED_TEST_SUITE(DummyTest, Types);

static void LoadDummyPlugin() {
  ::dali::PluginManager::LoadLibrary(DummyPluginLibPath());
}

TYPED_TEST(DummyTest, PluginShouldBeUsable) {
  LoadDummyPlugin();
  this->RunTest("CustomDummy");
}

}  // namespace other_ns
