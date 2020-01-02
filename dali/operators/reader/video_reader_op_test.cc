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
#include <cmath>

#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/nvml_wrap.h"

namespace dali {

class VideoReaderTest : public ::testing::Test {
 protected:
  std::vector<std::pair<std::string, std::string>> Outputs() {
    return {{"frames", "gpu"}};
  }
  std::vector<std::pair<std::string, std::string>> LabelledOutputs() {
    return {{"frames", "gpu"}, {"labels", "gpu"}};
  }
  std::vector<std::pair<std::string, std::string>> Output_frames_label_frame_num() {
    return {{"frames", "gpu"}, {"labels", "gpu"}, {"frame_num", "gpu"}};
  }
  std::vector<std::pair<std::string, std::string>> Output_frames_label_timestamp() {
    return {{"frames", "gpu"}, {"labels", "gpu"}, {"timestamp", "gpu"}};
  }
  std::vector<std::pair<std::string, std::string>> Output_frames_frame_num_timestamp() {
    return {{"frames", "gpu"}, {"labels", "gpu"}, {"frame_num", "gpu"}, {"timestamp", "gpu"}};
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
  float driverVersion = 0;
  char version[80];

  if (nvml::wrapSymbols() != DALISuccess) {
    FAIL() << "wrapSymbols() failed";
  }
  if (nvml::wrapNvmlInit() != DALISuccess) {
    FAIL() << "wrapNvmlInit() failed";
  }

  if (nvml::wrapNvmlSystemGetDriverVersion(version, sizeof version) != DALISuccess) {
    FAIL() << "wrapNvmlSystemGetDriverVersion failed!";
  }

  driverVersion = std::stof(version);

#if defined __powerpc64__
  std::cerr << "Test case running on powerpc64, driver version " << driverVersion << '\n';
  if (driverVersion < 415) {
#elif defined __x86_64__
  std::cerr << "Test case running on x86_64, driver version " << driverVersion << '\n';
  if (driverVersion < 396) {
#endif
    GTEST_SKIP() << "Test skipped because cuvidReconfigureDecoder API is not"
                    " supported by installed driver version";
  }

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
  pipe.Build(this->LabelledOutputs());

  pipe.RunCPU();
  pipe.RunGPU();
  pipe.Outputs(&ws);

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

TEST_F(VideoReaderTest, FrameLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("random_shuffle", false)
    .AddArg("sequence_length", sequence_length)
    .AddArg("enable_frame_num", true)
    .AddArg("image_type", DALI_YCbCr)
    .AddArg(
      "file_list", "/tmp/file_list.txt")
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu")
    .AddOutput("frame_num", "gpu"));

  pipe.Build(this->Output_frames_label_frame_num());

  DeviceWorkspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu, 0);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu, 0);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu, 0);
    cudaStreamSynchronize(0);

    const uint8 *frames = static_cast<const uint8 *>(frames_cpu.raw_data());
    const int *label = static_cast<const int *>(labels_cpu.raw_data());
    const int *frame_num = static_cast<const int *>(frame_num_cpu.raw_data());

    ASSERT_EQ(frames[0], frame_num[0]);
  }
}

TEST_F(VideoReaderTest, FrameLabelsWithFileListFrameNum) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("random_shuffle", false)
    .AddArg("sequence_length", sequence_length)
    .AddArg("enable_frame_num", true)
    .AddArg("enable_timestamps", true)
    .AddArg("file_list_frame_num", true)
    .AddArg("image_type", DALI_YCbCr)
    .AddArg(
      "file_list", "/tmp/file_list_frame_num.txt")
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu")
    .AddOutput("frame_num", "gpu")
    .AddOutput("timestamp", "gpu"));

  pipe.Build(this->Output_frames_frame_num_timestamp());

  DeviceWorkspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);
    const auto &timestamp_gpu = ws.Output<dali::GPUBackend>(3);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu, 0);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu, 0);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu, 0);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(timestamp_gpu, 0);
    cudaStreamSynchronize(0);

    const uint8 *frames = static_cast<const uint8 *>(frames_cpu.raw_data());
    const int *label = static_cast<const int *>(labels_cpu.raw_data());
    const int *frame_num = static_cast<const int *>(frame_num_cpu.raw_data());
    const double *timestamps = static_cast<const double *>(timestamps_cpu.raw_data());

    ASSERT_DOUBLE_EQ(frame_num[0], timestamps[0]*25);
    switch (label[0]) {
      case 0:
        ASSERT_TRUE(frame_num[0] >= 0 && frame_num[0] < 50);
        break;
      case 1:
        ASSERT_TRUE(frame_num[0] >= 50 && frame_num[0] < 100);
        break;
      case 2:
        ASSERT_TRUE(frame_num[0] >= 100 && frame_num[0] < 150);
        break;
      default:
        FAIL() << "Unexpected label";
    }
  }
}

TEST_F(VideoReaderTest, TimestampLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("random_shuffle", false)
    .AddArg("sequence_length", sequence_length)
    .AddArg("enable_timestamps", true)
    .AddArg("enable_frame_num", true)
    .AddArg("image_type", DALI_YCbCr)
    .AddArg(
      "file_list", "/tmp/file_list_timestamp.txt")
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu")
    .AddOutput("frame_num", "gpu")
    .AddOutput("timestamp", "gpu"));

  pipe.Build(this->Output_frames_frame_num_timestamp());

  DeviceWorkspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);
    const auto &timestamp_gpu = ws.Output<dali::GPUBackend>(3);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu, 0);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu, 0);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(timestamp_gpu, 0);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu, 0);
    cudaStreamSynchronize(0);

    const uint8 *frames = static_cast<const uint8 *>(frames_cpu.raw_data());
    const int *label = static_cast<const int *>(labels_cpu.raw_data());
    const int *frame_num = static_cast<const int *>(frame_num_cpu.raw_data());
    const double *timestamps = static_cast<const double *>(timestamps_cpu.raw_data());

    ASSERT_DOUBLE_EQ(frame_num[0], timestamps[0]*25);
  }
}

TEST_F(VideoReaderTest, StartEndLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("random_shuffle", false)
    .AddArg("sequence_length", sequence_length)
    .AddArg("enable_timestamps", true)
    .AddArg("image_type", DALI_YCbCr)
    .AddArg(
      "file_list", "/tmp/file_list.txt")
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu")
    .AddOutput("timestamp", "gpu"));

  pipe.Build(this->Output_frames_label_timestamp());

  DeviceWorkspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu, 0);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu, 0);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(frame_num_gpu, 0);
    cudaStreamSynchronize(0);

    const uint8 *frames = static_cast<const uint8 *>(frames_cpu.raw_data());
    const int *label = static_cast<const int *>(labels_cpu.raw_data());
    const double *timestamps = static_cast<const double *>(timestamps_cpu.raw_data());

    ASSERT_EQ(*label, std::floor(timestamps[0]/100));
  }
}

TEST_F(VideoReaderTest, MultipleFrameRates) {
  const int sequence_length = 1;
  const int iterations = 100;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
    OpSpec("VideoReader")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "file_root",
      testing::dali_extra_path() +
      "/db/video/multiple_framerate/")
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu"));

  pipe.Build(this->LabelledOutputs());

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

}  // namespace dali
