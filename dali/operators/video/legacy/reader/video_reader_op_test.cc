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
#include <cmath>
#include <cstring>
#include <fstream>
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"
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

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", 60)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/vfr_test.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  EXPECT_THROW(pipe.Build(this->Outputs()), std::runtime_error);
}

TEST_F(VideoReaderTest, VariableFrameRate2) {
  Pipeline pipe(1, 1, 0);

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", 60)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/vfr_test.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  pipe.Build(this->Outputs());
  EXPECT_THROW([&]() {
      for (int i = 0; i < 10; ++i) {
        pipe.Run();
        pipe.Outputs(&ws);
      }
    }(), std::runtime_error);
}


TEST_F(VideoReaderTest, FractionalConstantFrameRate) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  pipe.AddOperator(
      OpSpec("VideoReader")
          .AddArg("device", "gpu")
          .AddArg("sequence_length", sequence_length)
          .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                        "/db/video/cfr_ntsc_29_97_test.mp4"})
          .AddOutput("frames", StorageDevice::GPU));

  pipe.Build(this->Outputs());
}

TEST_F(VideoReaderTest, ConstantFrameRate) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/cfr_test.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  pipe.Build(this->Outputs());

  Workspace ws;
  pipe.Run();
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
#if defined(__powerpc64__) || defined(__x86_64__)
  float driverVersion = 0;

#if NVML_ENABLED
  auto nvml_handle = nvml::NvmlInstance::CreateNvmlInstance();
  driverVersion = nvml::GetDriverVersion();
#endif


#if defined(__powerpc64__)
  std::cerr << "Test case running on powerpc64, driver version " << driverVersion << '\n';
  if (driverVersion < 415)
#elif defined(__x86_64__)
  std::cerr << "Test case running on x86_64, driver version " << driverVersion << '\n';
  if (driverVersion < 396)
#endif
  {
    GTEST_SKIP() << "Test skipped because cuvidReconfigureDecoder API is not"
                    " supported by installed driver version";
  }
#endif

  Pipeline pipe(batch_size, 1, 0);

  pipe.AddOperator(
      OpSpec("VideoReader")
          .AddArg("device", "gpu")
          .AddArg("sequence_length", sequence_length)
          .AddArg("random_shuffle", true)
          .AddArg("initial_fill", initial_fill)
          .AddArg("file_root", std::string{testing::dali_extra_path() + "/db/video_resolution/"})
          .AddOutput("frames", StorageDevice::GPU)
          .AddOutput("labels", StorageDevice::GPU));

  Workspace ws;
  pipe.Build(this->LabelledOutputs());

  pipe.Run();
  pipe.Outputs(&ws);

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &labels_output = ws.Output<dali::GPUBackend>(1);

  TensorList<CPUBackend> labels_cpu;
  labels_cpu.Copy(labels_output);
  CUDA_CALL(cudaStreamSynchronize(0));
  labels_cpu.set_type<int>();

  for (int i = 0; i < batch_size; ++i) {
    const auto *labels = labels_cpu.tensor<int>(i);
    auto frames_shape = frames_output.tensor_shape(i);

    switch (labels[0]) {
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
          .AddArg("filenames",
                  std::vector<std::string>{testing::dali_extra_path() +
                                           "/db/video/ucf101_test/packed_bframes_test.avi"})
          .AddOutput("frames", StorageDevice::GPU));

  pipe.Build(this->Outputs());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_output = ws.Output<dali::GPUBackend>(0);
    const auto &frames_shape = frames_output.shape();

    ASSERT_EQ(frames_shape.size(), batch_size);
    ASSERT_EQ(frames_shape[0][0], sequence_length);
  }
}

inline bool IsUnsupportedCodec(const char *error_message) {
  const char *unsupported_codec_messages[] = {
      "Decoder hardware does not support this video codec and/or chroma format",
      "Unsupported Code"};
  for (const char *part : unsupported_codec_messages) {
    if (std::strstr(error_message, part)) return true;
  }
  return false;
}

TEST_F(VideoReaderTest, Vp9Profile0) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  // richer FFmpeg configuration leads to different behaviour of VFR heuristics so dissable it for
  // this video
  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/vp9/vp9_0.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  try {
    pipe.Build(this->Outputs());

    pipe.Run();
    pipe.Outputs(&ws);
  } catch (const std::exception &e) {
    if (IsUnsupportedCodec(e.what())) {
      GTEST_SKIP() << "Skipped because of unsupported codec. Original error:\n" << e.what();
    } else {
      throw;
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

TEST_F(VideoReaderTest, Vp9Profile2) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;
  const string unsupported_exception_msg =
      "Decoder hardware does not support this video codec"
      " and/or chroma format";

  // richer FFmpeg configuration leads to different behaviour of VFR heuristics so dissable it for
  // this video
  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/vp9/vp9_2.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  try {
    pipe.Build(this->Outputs());

    pipe.Run();
    pipe.Outputs(&ws);
  } catch (const std::exception &e) {
    if (IsUnsupportedCodec(e.what())) {
      GTEST_SKIP() << "Skipped because of unsupported codec. Original error:\n" << e.what();
    } else {
      throw;
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}


TEST_F(VideoReaderTest, Vp8Profile0) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;

  // richer FFmpeg configuration leads to different behaviour of VFR heuristics so dissable it for
  // this video
  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/vp8/vp8.webm"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  try {
    pipe.Build(this->Outputs());

    pipe.Run();
    pipe.Outputs(&ws);
  } catch (const std::exception &e) {
    if (IsUnsupportedCodec(e.what())) {
      GTEST_SKIP() << "Skipped because of unsupported codec. Original error:\n" << e.what();
    } else {
      throw;
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

TEST_F(VideoReaderTest, MJpeg) {
  Pipeline pipe(1, 1, 0);
  const int sequence_length = 60;
  const string unsupported_exception_msg =
      "Decoder hardware does not support this video codec"
      " and/or chroma format";

  // richer FFmpeg configuration leads to different behaviour of VFR heuristics so dissable it for
  // this video
  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                                     "/db/video/mjpeg/mjpeg.avi"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  try {
    pipe.Build(this->Outputs());

    pipe.Run();
    pipe.Outputs(&ws);
  } catch (const std::exception &e) {
    if (IsUnsupportedCodec(e.what())) {
      GTEST_SKIP() << "Skipped because of unsupported codec. Original error:\n" << e.what();
    } else {
      throw;
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 1);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

TEST_F(VideoReaderTest, HEVC) {
  Pipeline pipe(16, 1, 0);
  const int sequence_length = 3;
  const string unsupported_exception_msg =
      "Decoder hardware does not support this video codec"
      " and/or chroma format";

  // richer FFmpeg configuration leads to different behaviour of VFR heuristics so dissable it for
  // this video
  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("skip_vfr_check", true)
                       .AddArg("filenames", std::vector<std::string>
                             {testing::dali_extra_path() +"/db/video/hevc/sintel_trailer-720p.mp4"})
                       .AddOutput("frames", StorageDevice::GPU));

  Workspace ws;
  constexpr int iterations = 10;
  try {
    pipe.Build(this->Outputs());
    for (int i = 0; i < iterations; ++i) {
      pipe.Run();

      pipe.Outputs(&ws);
    }
  } catch (const std::exception &e) {
    if (IsUnsupportedCodec(e.what())) {
      GTEST_SKIP() << "Skipped because of unsupported codec. Original error:\n" << e.what();
    } else {
      throw;
    }
  }

  const auto &frames_output = ws.Output<dali::GPUBackend>(0);
  const auto &frames_shape = frames_output.shape();

  ASSERT_EQ(frames_shape.size(), 16);
  ASSERT_EQ(frames_shape[0][0], sequence_length);
}

TEST_F(VideoReaderTest, FrameLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  char file_list_path_template[] = "/tmp/video_reader_test_XXXXXX";
  int fd = mkstemp(file_list_path_template);
  close(fd);
  std::string file_list_path(file_list_path_template);
  auto cleanup = std::unique_ptr<void, std::function<void(void*)>>(
      nullptr, [file_list_path](void*) { std::remove(file_list_path.c_str()); });
  std::ofstream file_list(file_list_path);
  file_list << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 0 0 99\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 1 100 199\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 2 200 256\n";
  file_list.close();

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("random_shuffle", false)
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("enable_frame_num", true)
                       .AddArg("image_type", DALI_YCbCr)
                       .AddArg("file_list", file_list_path)
                       .AddOutput("frames", StorageDevice::GPU)
                       .AddOutput("labels", StorageDevice::GPU)
                       .AddOutput("frame_num", StorageDevice::GPU));

  pipe.Build(this->Output_frames_label_frame_num());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *frames = frames_cpu.tensor<uint8_t>(0);
    const auto *frame_num = frame_num_cpu.tensor<int>(0);

    ASSERT_EQ(frames[0], frame_num[0]);
  }
}

TEST_F(VideoReaderTest, FrameLabelsFilenames) {
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
          .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                        "/db/video/frame_num_timestamp/test.mp4"})
          .AddArg("labels", std::vector<int>{})
          .AddOutput("frames", StorageDevice::GPU)
          .AddOutput("labels", StorageDevice::GPU)
          .AddOutput("frame_num", StorageDevice::GPU));

  pipe.Build(this->Output_frames_label_frame_num());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *frames = frames_cpu.tensor<uint8_t>(0);
    const auto *label = labels_cpu.tensor<int>(0);
    const auto *frame_num = frame_num_cpu.tensor<int>(0);

    ASSERT_EQ(frames[0], frame_num[0]);
    ASSERT_EQ(label[0], 0);
  }
}

TEST_F(VideoReaderTest, LabelsFilenames) {
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
          .AddArg("filenames", std::vector<std::string>{testing::dali_extra_path() +
                                                        "/db/video/frame_num_timestamp/test.mp4"})
          .AddArg("labels", std::vector<int>{99})
          .AddOutput("frames", StorageDevice::GPU)
          .AddOutput("labels", StorageDevice::GPU)
          .AddOutput("frame_num", StorageDevice::GPU));

  pipe.Build(this->Output_frames_label_frame_num());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *frames = frames_cpu.tensor<uint8_t>(0);
    const auto *label = labels_cpu.tensor<int>(0);
    const auto *frame_num = frame_num_cpu.tensor<int>(0);

    ASSERT_EQ(frames[0], frame_num[0]);
    ASSERT_EQ(label[0], 99);
  }
}

TEST_F(VideoReaderTest, FrameLabelsWithFileListFrameNum) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  char file_list_path_template[] = "/tmp/video_reader_test_XXXXXX";
  int fd = mkstemp(file_list_path_template);
  close(fd);
  std::string file_list_path(file_list_path_template);
  auto cleanup = std::unique_ptr<void, std::function<void(void*)>>(
      nullptr, [file_list_path](void*) { std::remove(file_list_path.c_str()); });
  std::ofstream file_list(file_list_path);
  file_list << testing::dali_extra_path()
            << "/db/video/frame_num_timestamp/test_25fps.mp4 0 0 49\n"
            << testing::dali_extra_path()
            << "/db/video/frame_num_timestamp/test_25fps.mp4 1 50 99\n"
            << testing::dali_extra_path()
            << "/db/video/frame_num_timestamp/test_25fps.mp4 2 100 149\n";
  file_list.close();

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("random_shuffle", false)
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("enable_frame_num", true)
                       .AddArg("enable_timestamps", true)
                       .AddArg("file_list_frame_num", true)
                       .AddArg("image_type", DALI_YCbCr)
                       .AddArg("file_list", file_list_path)
                       .AddOutput("frames", StorageDevice::GPU)
                       .AddOutput("labels", StorageDevice::GPU)
                       .AddOutput("frame_num", StorageDevice::GPU)
                       .AddOutput("timestamp", StorageDevice::GPU));

  pipe.Build(this->Output_frames_frame_num_timestamp());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);
    const auto &timestamp_gpu = ws.Output<dali::GPUBackend>(3);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(timestamp_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *label = labels_cpu.tensor<int>(0);
    const auto *frame_num = frame_num_cpu.tensor<int>(0);
    const auto *timestamps = timestamps_cpu.tensor<double>(0);

    ASSERT_DOUBLE_EQ(frame_num[0], timestamps[0] * 25);
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

  // Cleanup temporary files
  std::remove(file_list_path.c_str());
}

TEST_F(VideoReaderTest, TimestampLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  char file_list_path_template[] = "/tmp/video_reader_test_XXXXXX";
  int fd = mkstemp(file_list_path_template);
  close(fd);
  std::string file_list_path(file_list_path_template);
  auto cleanup = std::unique_ptr<void, std::function<void(void*)>>(
      nullptr, [file_list_path](void*) { std::remove(file_list_path.c_str()); });
  std::ofstream file_list(file_list_path);
  file_list << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test_25fps.mp4 0 0 1\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test_25fps.mp4 1 2 3\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test_25fps.mp4 2 4 5\n";
  file_list.close();

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("random_shuffle", false)
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("enable_timestamps", true)
                       .AddArg("enable_frame_num", true)
                       .AddArg("image_type", DALI_YCbCr)
                       .AddArg("file_list", file_list_path)
                       .AddOutput("frames", StorageDevice::GPU)
                       .AddOutput("labels", StorageDevice::GPU)
                       .AddOutput("frame_num", StorageDevice::GPU)
                       .AddOutput("timestamp", StorageDevice::GPU));

  pipe.Build(this->Output_frames_frame_num_timestamp());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);
    const auto &timestamp_gpu = ws.Output<dali::GPUBackend>(3);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(timestamp_gpu);
    TensorList<CPUBackend> frame_num_cpu;
    frame_num_cpu.Copy(frame_num_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *frame_num = frame_num_cpu.tensor<int>(0);
    const auto *timestamps = timestamps_cpu.tensor<double>(0);

    ASSERT_DOUBLE_EQ(frame_num[0], timestamps[0] * 25);
  }

  // Cleanup temporary files
  std::remove(file_list_path.c_str());
}

TEST_F(VideoReaderTest, StartEndLabels) {
  const int sequence_length = 1;
  const int iterations = 256;
  const int batch_size = 1;
  Pipeline pipe(batch_size, 1, 0);

  char file_list_path_template[] = "/tmp/video_reader_test_XXXXXX";
  int fd = mkstemp(file_list_path_template);
  close(fd);
  std::string file_list_path(file_list_path_template);
  auto cleanup = std::unique_ptr<void, std::function<void(void*)>>(
      nullptr, [file_list_path](void*) { std::remove(file_list_path.c_str()); });
  std::ofstream file_list(file_list_path);
  file_list << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 0 0 99\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 1 100 199\n"
            << testing::dali_extra_path() << "/db/video/frame_num_timestamp/test.mp4 2 200 256\n";
  file_list.close();

  pipe.AddOperator(OpSpec("VideoReader")
                       .AddArg("device", "gpu")
                       .AddArg("random_shuffle", false)
                       .AddArg("sequence_length", sequence_length)
                       .AddArg("enable_timestamps", true)
                       .AddArg("image_type", DALI_YCbCr)
                       .AddArg("file_list", file_list_path)
                       .AddOutput("frames", StorageDevice::GPU)
                       .AddOutput("labels", StorageDevice::GPU)
                       .AddOutput("timestamp", StorageDevice::GPU));

  pipe.Build(this->Output_frames_label_timestamp());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_gpu = ws.Output<dali::GPUBackend>(0);
    const auto &label_gpu = ws.Output<dali::GPUBackend>(1);
    const auto &frame_num_gpu = ws.Output<dali::GPUBackend>(2);

    TensorList<CPUBackend> frames_cpu;
    frames_cpu.Copy(frames_gpu);
    TensorList<CPUBackend> labels_cpu;
    labels_cpu.Copy(label_gpu);
    TensorList<CPUBackend> timestamps_cpu;
    timestamps_cpu.Copy(frame_num_gpu);
    CUDA_CALL(cudaStreamSynchronize(0));

    const auto *label = labels_cpu.tensor<int>(0);
    const auto *timestamps = timestamps_cpu.tensor<double>(0);

    ASSERT_EQ(*label, std::floor(timestamps[0] / 100));
  }

  std::remove(file_list_path.c_str());
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
          .AddArg("file_root", testing::dali_extra_path() + "/db/video/multiple_framerate/")
          .AddOutput("frames", StorageDevice::GPU)
          .AddOutput("labels", StorageDevice::GPU));

  pipe.Build(this->LabelledOutputs());

  Workspace ws;
  for (int i = 0; i < iterations; ++i) {
    pipe.Run();
    pipe.Outputs(&ws);
    const auto &frames_output = ws.Output<dali::GPUBackend>(0);
    const auto &frames_shape = frames_output.shape();

    ASSERT_EQ(frames_shape.size(), batch_size);
    ASSERT_EQ(frames_shape[0][0], sequence_length);
  }
}

}  // namespace dali
