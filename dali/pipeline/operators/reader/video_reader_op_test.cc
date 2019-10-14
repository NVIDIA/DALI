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
  std::vector<std::pair<std::string, std::string>> LabelledOutputs() {
    return {{"frames", "gpu"}, {"labels", "gpu"}};
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

TEST_F(VideoReaderTest, FractionalConstantFrameRate) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() + "/db/video/cfr_ntsc_29_97_test.mp4"})
    .AddOutput("frames", "gpu"));

  pipe.Build(this->Outputs());
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

TEST_F(VideoReaderTest, MultipleVideoResolution) {
  const int batch_size = 10;
  const int sequence_length = 1;
  const int initial_fill = 10;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("random_shuffle", true)
    .AddArg("initial_fill", initial_fill)
    .AddArg(
      "file_root",
      std::string{testing::dali_extra_path() + "/db/video_resolution/"})
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu"));


  DeviceWorkspace ws;
  try {
    pipe.Build(this->LabelledOutputs());

    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  } catch (unsupported_exception &) {
    GTEST_SKIP() << "Test skipped because cuvidReconfigureDecoder API is not"
                    " supported by installed driver version";
  } catch (...) {
    FAIL();
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &labels_output = ws.Output<dali::GPUBackend>(1);

  TensorList<CPUBackend> labels_cpu;
  labels_cpu.Copy(labels_output, 0);
  cudaStreamSynchronize(0);
  labels_cpu.set_type(TypeInfo::Create<int>());
  const int *labels = static_cast<const int *>(labels_cpu.raw_data());


  for (int i =0; i < batch_size; ++i) {
    auto frames_shape = frames_output.tensor_shape(i);

    switch (labels[i]) {
    case 0:
      ASSERT_EQ(frames_shape[1], 2160);
      ASSERT_EQ(frames_shape[2], 3840);
      break;
    case 1:
      ASSERT_EQ(frames_shape[1], 1080);
      ASSERT_EQ(frames_shape[2], 1920);
      break;
    case 2:
      ASSERT_EQ(frames_shape[1], 240);
      ASSERT_EQ(frames_shape[2], 334);
      break;
    default:
      FAIL() << "Unexpected label";
    }
  }
}

TEST_F(VideoReaderTest, PackedBFrames) {
  const int sequence_length = 5;
  const int iterations = 10;
  const int batch_size = 5;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() +
      "/db/video/ucf101_test/packed_bframes_test.avi"})
    .AddOutput("frames", "gpu"));

  pipe.Build(this->Outputs());

  DeviceWorkspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    const auto &frames_output = ws.Output<dali::GPUBackend>(0);
    const auto &frames_shape = frames_output.shape();

    ASSERT_EQ(frames_shape.size(), batch_size);
    ASSERT_EQ(frames_shape[0][0], sequence_length);
  }
}

TEST_F(VideoReaderTest, Vp9Profile0) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() + "/db/video/vp9/vp9_0.mp4"})
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

TEST_F(VideoReaderTest, Vp9Profile2) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;
  const string unsupported_exception_msg = "Decoder hardware does not support this video codec"
                                          " and/or chroma format";

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{testing::dali_extra_path() + "/db/video/vp9/vp9_2.mp4"})
    .AddOutput("frames", "gpu"));

  DeviceWorkspace ws;
  try {
    pipe.Build(this->Outputs());

    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
  } catch (std::exception &e) {
    if (string(e.what()).find(unsupported_exception_msg) != std::string::npos) {
      GTEST_SKIP() << "Test skipped because VP9 codec with 10/12 bit depth is not supported"
                      "on this hardware";
    } else {
    FAIL();
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

}  // namespace dali
