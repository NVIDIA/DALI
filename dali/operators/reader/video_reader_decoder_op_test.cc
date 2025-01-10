// Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/cv_mat_utils.h"
#include "dali/core/dev_buffer.h"


namespace dali {
class VideoReaderDecoderBaseTest : public VideoTestBase {
 public:
  template<typename Backend>
  void RunTest(
    std::vector<std::string> &videos_paths,
    std::vector<TestVideo> &ground_truth_videos);

  template<typename Backend>
  void RunShuffleTest();

  virtual void AssertLabel(const int *label, int ground_truth_label) = 0;

  virtual void AssertFrame(
    int frame_id, const uint8_t *frame, TestVideo &ground_truth) = 0;

  template<typename Backend>
  int GetFrameIdx(dali::TensorList<Backend> &device_frame_idx);

 private:
  template<typename Backend>
  void RunTestImpl(
        std::vector<std::string> &videos_paths,
        std::vector<TestVideo> &ground_truth_videos,
        std::string backend,
        int device_id) {
    const int batch_size = 4;
    const int sequence_length = 6;
    const int stride = 3;
    const int step = 10;

    Pipeline pipe(batch_size, 4, device_id);

    auto storage_dev = ParseStorageDevice(backend);
    pipe.AddOperator(OpSpec("experimental__readers__Video")
      .AddArg("device", backend)
      .AddArg("sequence_length", sequence_length)
      .AddArg("stride", stride)
      .AddArg("step", step)
      .AddArg(
        "filenames",
        videos_paths)
      .AddArg("labels", std::vector<int>{0, 1})
      .AddOutput("frames", storage_dev)
      .AddOutput("labels", storage_dev));

    pipe.Build({{"frames", backend}, {"labels", backend}});

    int num_sequences = 20;
    int sequence_id = 0;
    int batch_id = 0;
    int gt_frame_id = 0;

    int video_idx = 0;

    Workspace ws;
    while (sequence_id < num_sequences) {
      pipe.Run();
      pipe.Outputs(&ws);

      auto &frame_video_output = ws.Output<Backend>(0);
      auto &frame_label_output = ws.Output<Backend>(1);

      ASSERT_EQ(frame_video_output.GetLayout(), "FHWC");

      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        const auto sample = frame_video_output.template tensor<uint8_t>(sample_id);
        const auto label = frame_label_output.template tensor<int>(sample_id);

        AssertLabel(label, video_idx);

        for (int i = 0; i < sequence_length; ++i) {
          AssertFrame(
            gt_frame_id + i * stride,
            sample + i *  ground_truth_videos[video_idx].FrameSize(),
            ground_truth_videos[video_idx]);
        }

        gt_frame_id += step;
        ++sequence_id;

        if (gt_frame_id + stride * sequence_length >= ground_truth_videos[video_idx].NumFrames()) {
          gt_frame_id = 0;
          ++video_idx;
          if (video_idx == this->NumVideos()) {
            video_idx = 0;
          }
        }
      }
      ++batch_id;
    }
  }

  template<typename Backend>
  void RunShuffleTestImpl(
        std::string backend,
        int device_id) {
    const int batch_size = 1;
    const int sequence_length = 1;
    const int seed = 1;

    auto &ground_truth_video = cfr_videos_[0];

    Pipeline pipe(batch_size, 1, device_id, seed);

    auto storage_device = ParseStorageDevice(backend);
    pipe.AddOperator(OpSpec("experimental__readers__Video")
      .AddArg("device", backend)
      .AddArg("sequence_length", sequence_length)
      .AddArg("random_shuffle", true)
      .AddArg("enable_frame_num", true)
      .AddArg("initial_fill", cfr_videos_[0].NumFrames())
      .AddArg(
        "filenames",
        std::vector<std::string>{cfr_videos_paths_[0]})
      .AddOutput("frames", storage_device)
      .AddOutput("frame_idx", storage_device));

    pipe.Build({{"frames", backend}, {"frame_idx", backend}});

    int num_sequences = 5;

    Workspace ws;
    for (int sequence_id = 0; sequence_id < num_sequences; ++sequence_id) {
      pipe.Run();
      pipe.Outputs(&ws);

      auto &frame_video_output = ws.Output<Backend>(0);
      const auto sample = frame_video_output.template tensor<uint8_t>(0);
      int frame_idx = GetFrameIdx(ws.Output<Backend>(1));

      // We want to access correct order, so we compare only the first frame of the sequence
      AssertFrame(frame_idx, sample, ground_truth_video);
    }
  }
};

template<>
void VideoReaderDecoderBaseTest::RunTest<dali::CPUBackend>(
  std::vector<std::string> &videos_paths,
  std::vector<TestVideo> &ground_truth_videos) {
    RunTestImpl<dali::CPUBackend>(
      videos_paths, ground_truth_videos, "cpu", dali::CPU_ONLY_DEVICE_ID);
}

template<>
void VideoReaderDecoderBaseTest::RunShuffleTest<dali::CPUBackend>() {
    RunShuffleTestImpl<dali::CPUBackend>("cpu", dali::CPU_ONLY_DEVICE_ID);
}

template<>
int VideoReaderDecoderBaseTest::GetFrameIdx(
  dali::TensorList<dali::CPUBackend> &device_frame_idx) {
    const auto frame_idx = device_frame_idx.template tensor<int>(0);
    int frame_idx_buffer = -1;
    std::copy_n(frame_idx, 1, &frame_idx_buffer);
    return frame_idx_buffer;
}

template<>
void VideoReaderDecoderBaseTest::RunTest<dali::GPUBackend>(
  std::vector<std::string> &videos_paths,
  std::vector<TestVideo> &ground_truth_videos) {
    RunTestImpl<dali::GPUBackend>(
      videos_paths, ground_truth_videos, "gpu", 0);
}

template<>
void VideoReaderDecoderBaseTest::RunShuffleTest<dali::GPUBackend>() {
    RunShuffleTestImpl<dali::GPUBackend>("gpu", 0);
}

template<>
int VideoReaderDecoderBaseTest::GetFrameIdx(
  dali::TensorList<dali::GPUBackend> &device_frame_idx) {
    const auto frame_idx = device_frame_idx.template tensor<int>(0);
    int frame_idx_buffer = -1;
    MemCopy(&frame_idx_buffer, frame_idx, sizeof(int));
    return frame_idx_buffer;
}

class VideoReaderDecoderCpuTest : public VideoReaderDecoderBaseTest {
 public:
  void AssertLabel(const int *label, int ground_truth_label) override {
    ASSERT_EQ(label[0], ground_truth_label);
  }

  void AssertFrame(int frame_id, const uint8_t *frame, TestVideo &ground_truth) override {
    ground_truth.CompareFrameAvgError(frame_id, frame);
  }
};

class VideoReaderDecoderGpuTest : public VideoReaderDecoderBaseTest {
 public:
  void SetUp() override {
    frame_buffer_.resize(MaxFrameSize());
  }

  void AssertLabel(const int *label, int ground_truth_label) override {
    label_buffer_ = -1;
    MemCopy(&label_buffer_, label, sizeof(DALIDataType::DALI_INT32));
    ASSERT_EQ(label_buffer_, ground_truth_label);
  }

  void AssertFrame(int frame_id, const uint8_t *frame, TestVideo &ground_truth) override {
    frame_buffer_.clear();
    MemCopy(frame_buffer_.data(), frame, ground_truth.FrameSize());
    ground_truth.CompareFrameAvgError(frame_id, frame_buffer_.data(), 1.5);
  }

 private:
  int label_buffer_ = -1;
  std::vector<uint8_t> frame_buffer_;
};

TEST_F(VideoReaderDecoderCpuTest, ConstantFrameRate_CpuOnlyTests) {
  RunTest<dali::CPUBackend>(cfr_videos_paths_, cfr_videos_);
}

TEST_F(VideoReaderDecoderCpuTest, VariableFrameRate_CpuOnlyTests) {
  RunTest<dali::CPUBackend>(vfr_videos_paths_, vfr_videos_);
}

TEST_F(VideoReaderDecoderCpuTest, LabelMismatch_CpuOnlyTests) {
  std::vector<std::string> paths {cfr_hevc_videos_paths_[0]};

  RunFailureTest([&]() -> void {
    RunTest<dali::CPUBackend>(paths, cfr_videos_);},
    "Current pipeline object is no longer valid.");
}

TEST_F(VideoReaderDecoderGpuTest, ConstantFrameRate) {
  RunTest<dali::GPUBackend>(cfr_videos_paths_, cfr_videos_);
}

TEST_F(VideoReaderDecoderGpuTest, VariableFrameRate) {
  RunTest<dali::GPUBackend>(vfr_videos_paths_, vfr_videos_);
}

TEST_F(VideoReaderDecoderCpuTest, RandomShuffle_CpuOnlyTests) {
  RunShuffleTest<dali::CPUBackend>();
}

TEST_F(VideoReaderDecoderGpuTest, RandomShuffle) {
  RunShuffleTest<dali::CPUBackend>();
}

class VideoReaderDecoderCompareTest : public VideoTestBase {};

TEST_F(VideoReaderDecoderCompareTest, CompareReaders) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  const int step = 10;
  const int shard_id = 3;
  const int num_shards = 10;
  const int seed = 1234;
  const int initial_fill = 50;

  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg("shard_id ", shard_id)
    .AddArg("num_shards ", num_shards)
    .AddArg("seed", seed)
    .AddArg("initial_fill", initial_fill)
    .AddArg("random_shuffle", true)
    .AddArg(
      "filenames",
      cfr_videos_paths_)
    .AddArg("labels", std::vector<int>{0, 1})
    .AddOutput("frames_cpu", StorageDevice::CPU)
    .AddOutput("labels_cpu", StorageDevice::CPU));
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg("shard_id ", shard_id)
    .AddArg("num_shards ", num_shards)
    .AddArg("seed", seed)
    .AddArg("initial_fill", initial_fill)
    .AddArg("random_shuffle", true)
    .AddArg(
      "filenames",
      cfr_videos_paths_)
    .AddArg("labels", std::vector<int>{0, 1})
    .AddOutput("frames_gpu", StorageDevice::GPU)
    .AddOutput("labels_gpu", StorageDevice::GPU));
  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg("shard_id ", shard_id)
    .AddArg("num_shards ", num_shards)
    .AddArg("seed", seed)
    .AddArg("initial_fill", initial_fill)
    .AddArg("random_shuffle", true)
    .AddArg(
      "filenames",
      cfr_videos_paths_)
    .AddArg("labels", std::vector<int>{0, 1})
    .AddOutput("frames_old", StorageDevice::GPU)
    .AddOutput("labels_old", StorageDevice::GPU));

  pipe.Build({
    {"frames_cpu", "cpu"},
    {"frames_gpu", "gpu"},
    {"frames_old", "gpu"},
    {"labels_cpu", "cpu"},
    {"labels_gpu", "gpu"},
    {"labels_old", "gpu"}});

  // Buffers on the CPU for outputs generated on the GPU
  vector<uint8_t> frame_buffer;
  frame_buffer.reserve(MaxFrameSize());
  int label_buffer = -1;

  for (int batch_id = 0; batch_id < 20; ++batch_id) {
    Workspace ws;
    pipe.Run();
    pipe.Outputs(&ws);

    auto &cpu_frame_output = ws.Output<dali::CPUBackend>(0);
    auto &gpu_frame_output = ws.Output<dali::GPUBackend>(1);
    auto &old_frame_output = ws.Output<dali::GPUBackend>(2);

    auto &cpu_label_output = ws.Output<dali::CPUBackend>(3);
    auto &gpu_label_output = ws.Output<dali::GPUBackend>(4);
    auto &old_label_output = ws.Output<dali::GPUBackend>(5);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      const auto cpu_sample = cpu_frame_output.tensor<uint8_t>(sample_id);
      const auto gpu_sample = gpu_frame_output.tensor<uint8_t>(sample_id);
      const auto old_sample = old_frame_output.tensor<uint8_t>(sample_id);

      const auto cpu_label = cpu_label_output.tensor<int>(sample_id);
      const auto gpu_label = gpu_label_output.tensor<int>(sample_id);
      const auto old_label = old_label_output.tensor<int>(sample_id);

      auto cpu_sample_shape = cpu_frame_output.tensor_shape(sample_id);
      auto gpu_sample_shape = gpu_frame_output.tensor_shape(sample_id);
      auto old_sample_shape = old_frame_output.tensor_shape(sample_id);

      ASSERT_EQ(cpu_sample_shape[0], sequence_length);
      ASSERT_EQ(cpu_sample_shape, gpu_sample_shape);
      ASSERT_EQ(cpu_sample_shape, old_sample_shape);

      int frame_size = cpu_sample_shape[1] * cpu_sample_shape[2] * cpu_sample_shape[3];

      label_buffer = -1;
      MemCopy(&label_buffer, gpu_label, sizeof(int));
      ASSERT_EQ(cpu_label[0], label_buffer);

      label_buffer = -1;
      MemCopy(&label_buffer, old_label, sizeof(int));
      ASSERT_EQ(cpu_label[0], label_buffer);

      frame_buffer.resize(frame_size);

      // ARM implementations of decoding work slightly different, so we need to adjust the eps
      double eps = 1.5;

      for (int i = 0; i < sequence_length; ++i) {
        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          gpu_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        dali::CompareFrameAvgError(
          cpu_sample + i * frame_size,
          frame_buffer.data(),
          frame_size,
          eps);

        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          old_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        dali::CompareFrameAvgError(
          cpu_sample + i * frame_size,
          frame_buffer.data(),
          frame_size,
          eps);
      }
    }
  }
}
}  // namespace dali
