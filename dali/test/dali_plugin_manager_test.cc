// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/pipeline/pipeline.h"

static const std::string& DummyPluginLibPath() {
  static const std::string plugin_lib =
      dali::test::CurrentExecutableDir() + "/libcustomdummyplugin.so";
  return plugin_lib;
}

namespace other_ns {

class DummyTest : public ::dali::DALITest {
};

static void LoadDummyPlugin() {
  ::dali::PluginManager::LoadLibrary(DummyPluginLibPath());
}

TEST_F(DummyTest, PluginShouldBeUsable) {
  LoadDummyPlugin();

  dali::Pipeline pipe(3, 1, 0);

  dali::TensorList<dali::CPUBackend> data;
  dali::test::MakeRandomBatch(data, 3);

  pipe.AddExternalInput("data");
  pipe.AddOperator(
      dali::OpSpec("CustomDummy")
      .AddArg("device", "cpu")
      .AddInput("data", "cpu")
      .AddOutput("out", "cpu"));
  std::vector<std::pair<std::string, std::string>> outputs = {{"out", "cpu"}};

  pipe.Build(outputs);
  pipe.SetExternalInput("data", data);
  dali::DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);
}

}  // namespace other_ns
