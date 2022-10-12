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
 public:
  static void LoadDummyPlugin() {
    ::dali::PluginManager::LoadLibrary(DummyPluginLibPath());
  }

  void TestPlugin(const std::string &backend) {
    LoadDummyPlugin();

    dali::TensorList<dali::CPUBackend> data;
    dali::test::MakeRandomBatch(data, 3);

    dali::Pipeline pipe(3, 1, 0);
    pipe.AddExternalInput("data");
    pipe.AddOperator(
        dali::OpSpec("CustomDummy")
        .AddArg("device", backend)
        .AddInput("data", backend)
        .AddOutput("out", backend));
    std::vector<std::pair<std::string, std::string>> outputs = {{"out", backend}};

    pipe.Build(outputs);
    pipe.SetExternalInput("data", data);
    dali::Workspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    dali::test::CheckResults(ws, 3, 0, data);
  }
};


TEST_F(DummyTest, TestPluginCPU) {
  this->TestPlugin("cpu");
}

TEST_F(DummyTest, TestPluginGPU) {
  this->TestPlugin("gpu");
}

}  // namespace other_ns
