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

#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"

namespace dali {

class VideoReaderTest : public ::testing::Test {
 protected:
  std::vector<std::pair<std::string, std::string>> Outputs() {
    return {{"frames", "gpu"}};
  }
};

TEST_F(VideoReaderTest, VariableFrameRate) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", 60)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() + "/db/video/vfr_test.mp4"})
    .AddOutput("frames", "gpu"));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(VideoReaderTest, ConstantFrameRate) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() + "/db/video/cfr_test.mp4"})
    .AddOutput("frames", "gpu"));

  pipe.Build(this->Outputs());

  DeviceWorkspace ws;
  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

}  // namespace dali
