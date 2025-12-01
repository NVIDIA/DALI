// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace other_ns {

const std::string &DummyPluginLibPath() {
  static const std::string plugin_lib =
      dali::test::CurrentExecutableDir() + "/libdali_customdummyplugin.so";
  return plugin_lib;
}

const std::string &DummyPluginLibPathGlobal() {
  static const std::string plugin_lib = "libdali_customdummyplugin.so";
  return plugin_lib;
}

void LoadDummyPlugin() {
  try {
    ::dali::PluginManager::LoadLibrary(DummyPluginLibPath());
  } catch (dali::DALIException &) {
    ::dali::PluginManager::LoadLibrary(DummyPluginLibPathGlobal());
  }
}

void TestPlugin(const std::string& backend) {
  dali::TensorList<dali::CPUBackend> data;
  dali::test::MakeRandomBatch(data, 3);

  dali::Pipeline pipe(3, 1, 0);
  pipe.AddExternalInput("data");
  auto storage_dev = dali::ParseStorageDevice(backend);
  pipe.AddOperator(dali::OpSpec("CustomDummy")
                       .AddArg("device", backend)
                       .AddInput("data", storage_dev)
                       .AddOutput("out", storage_dev));
  std::vector<std::pair<std::string, std::string>> outputs = {{"out", backend}};

  pipe.Build(outputs);
  pipe.SetExternalInput("data", data);
  dali::Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);

  dali::test::CheckResults(ws, 3, 0, data);

  _exit(0);  // This is to force each test to run in a separate process
}

TEST(DummyTest, TestPluginCPU) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  LoadDummyPlugin();
  // This is crucial so that each test case has a chance to load the plugin (new process).
  EXPECT_EXIT(TestPlugin("cpu"), testing::ExitedWithCode(0), "");
}

TEST(DummyTest, TestPluginGPU) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  LoadDummyPlugin();
  // This is crucial so that each test case has a chance to load the plugin (new process).
  EXPECT_EXIT(TestPlugin("gpu"), testing::ExitedWithCode(0), "");
}

TEST(DummyTest, LoadDirectory) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ::dali::PluginManager::LoadDirectory(dali::test::CurrentExecutableDir());
  // in conda we place plugins into main lib director, not app specific
  ::dali::PluginManager::LoadDirectory(dali::test::DefaultGlobalLibPath());
  // This is crucial so that each test case has a chance to load the plugin (new process).
  EXPECT_EXIT(TestPlugin("cpu"), testing::ExitedWithCode(0), "");
}

}  // namespace other_ns
