// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
};

class VideoReaderDecoderCpuTest : public VideoReaderDecoderBaseTest {
 public:
  void RunTest(
    std::vector<std::string> &videos_paths, std::vector<TestVideo> &ground_truth_videos) {
    const int batch_size = 4;
    const int sequence_length = 6;
    const int stride = 3;
    const int step = 10;

    Pipeline pipe(batch_size, 4, dali::CPU_ONLY_DEVICE_ID);

    pipe.AddOperator(OpSpec("experimental__readers__Video")
      .AddArg("device", "cpu")
      .AddArg("sequence_length", sequence_length)
      .AddArg("stride", stride)
      .AddArg("step", step)
      .AddArg(
        "filenames",
        videos_paths)
      .AddArg("labels", std::vector<int>{0, 1})
      .AddOutput("frames", "cpu")
      .AddOutput("labels", "cpu"));

    pipe.Build({{"frames", "cpu"}, {"labels", "cpu"}});

    int num_sequences = 20;
    int sequence_id = 0;
    int batch_id = 0;
    int gt_frame_id = 0;

    int video_idx = 0;

    while (sequence_id < num_sequences) {
      DeviceWorkspace ws;
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws);

      auto &frame_video_output = ws.template Output<dali::CPUBackend>(0);
      auto &frame_label_output = ws.template Output<dali::CPUBackend>(1);

      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        const auto sample = frame_video_output.tensor<uint8_t>(sample_id);
        const auto label = frame_label_output.tensor<int>(sample_id);

        ASSERT_EQ(label[0], video_idx);

        for (int i = 0; i < sequence_length; ++i) {
          ground_truth_videos[video_idx].CompareFrame(
            gt_frame_id + i * stride,
            sample + i * ground_truth_videos[video_idx].FrameSize());
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
};

class VideoReaderDecoderGpuTest : public VideoReaderDecoderBaseTest {
 public:
  void RunTest(
    std::vector<std::string> &videos_paths, std::vector<TestVideo> &ground_truth_videos) {
    const int batch_size = 4;
    const int sequence_length = 6;
    const int stride = 3;
    const int step = 10;

    Pipeline pipe(batch_size, 4, 0);

    pipe.AddOperator(OpSpec("experimental__readers__Video")
      .AddArg("device", "gpu")
      .AddArg("sequence_length", sequence_length)
      .AddArg("stride", stride)
      .AddArg("step", step)
      .AddArg(
        "filenames",
        videos_paths)
      .AddArg("labels", std::vector<int>{0, 1})
      .AddOutput("frames", "gpu")
      .AddOutput("labels", "gpu"));

    pipe.Build({{"frames", "gpu"}, {"labels", "gpu"}});

    int num_sequences = 20;
    int sequence_id = 0;
    int batch_id = 0;
    int gt_frame_id = 0;

    int video_idx = 0;

    std::vector<uint8_t> frame_cpu(MaxFrameSize());

    DeviceWorkspace ws;
    while (sequence_id < num_sequences) {
      pipe.RunCPU();
      pipe.RunGPU();
      pipe.Outputs(&ws);

      auto &frame_video_output = ws.template Output<dali::GPUBackend>(0);
      auto &frame_label_output = ws.template Output<dali::GPUBackend>(1);

      ASSERT_EQ(frame_video_output.GetLayout(), "FHWC");

      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        const auto sample = frame_video_output.tensor<uint8_t>(sample_id);
        const auto label = frame_label_output.tensor<int>(sample_id);

        int label_cpu = -1;
        MemCopy(
          &label_cpu, label, sizeof(DALIDataType::DALI_INT32));

        ASSERT_EQ(label_cpu, video_idx);

        for (int i = 0; i < sequence_length; ++i) {
          MemCopy(
            frame_cpu.data(),
            sample + i *  ground_truth_videos[video_idx].FrameSize(),
            ground_truth_videos[video_idx].FrameSize());

          ground_truth_videos[video_idx].CompareFrameAvgError(
            gt_frame_id + i * stride,
            frame_cpu.data());
        }

        gt_frame_id += step;
        ++sequence_id;

        if (gt_frame_id + stride * sequence_length >= vfr_videos_[video_idx].NumFrames()) {
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
};

TEST_F(VideoReaderDecoderCpuTest, CpuConstantFrameRate_CpuOnlyTests) {
  RunTest(cfr_videos_paths_, cfr_videos_);
}

TEST_F(VideoReaderDecoderCpuTest, CpuVariableFrameRate_CpuOnlyTests) {
  RunTest(vfr_videos_paths_, vfr_videos_);
}

TEST_F(VideoReaderDecoderGpuTest, GpuConstantFrameRate) {
  RunTest(cfr_videos_paths_, cfr_videos_);
}

TEST_F(VideoReaderDecoderGpuTest, GpuVariableFrameRate) {
  RunTest(vfr_videos_paths_, vfr_videos_);
}

TEST_F(VideoReaderDecoderCpuTest, RandomShuffle_CpuOnlyTests) {
  const int batch_size = 1;
  const int sequence_length = 1;
  const int seed = 1;

  auto &ground_truth_video = cfr_videos_[0];

  Pipeline pipe(batch_size, 1, dali::CPU_ONLY_DEVICE_ID, seed);

  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("random_shuffle", true)
    .AddArg("initial_fill", cfr_videos_[0].NumFrames())
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4"})
    .AddOutput("frames", "cpu"));

  pipe.Build({{"frames", "cpu"}});

  std::vector<int> expected_order = {29, 46, 33, 6, 37};

  int num_sequences = 5;

  for (int sequence_id = 0; sequence_id < num_sequences; ++sequence_id) {
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    auto &frame_video_output = ws.template Output<dali::CPUBackend>(0);
    const auto sample = frame_video_output.tensor<uint8_t>(0);

    // We want to access correct order, so we comapre only the first frame of the sequence
    ground_truth_video.CompareFrame(expected_order[sequence_id], sample);
  }
}

TEST_F(VideoReaderDecoderGpuTest, RandomShuffle) {
  const int batch_size = 1;
  const int sequence_length = 1;
  const int seed = 1;

  auto &ground_truth_video = cfr_videos_[0];

  std::vector<uint8_t> frame_cpu(ground_truth_video.FrameSize());

  Pipeline pipe(batch_size, 1, 0, seed);

  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("random_shuffle", true)
    .AddArg("initial_fill", cfr_videos_[0].NumFrames())
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4"})
    .AddOutput("frames", "gpu"));

  pipe.Build({{"frames", "gpu"}});

  std::vector<int> expected_order = {29, 46, 33, 6, 37};

  int num_sequences = 5;

  for (int sequence_id = 0; sequence_id < num_sequences; ++sequence_id) {
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    auto &frame_video_output = ws.template Output<dali::GPUBackend>(0);
    const auto sample = frame_video_output.tensor<uint8_t>(0);

    // We want to access correct order, so we comapre only the first frame of the sequence
    MemCopy(frame_cpu.data(), sample, ground_truth_video.FrameSize());
    ground_truth_video.CompareFrameAvgError(expected_order[sequence_id], frame_cpu.data());
  }
}

TEST_F(VideoReaderDecoderBaseTest, CompareReaders) {
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
    .AddOutput("frames_cpu", "cpu")
    .AddOutput("labels_cpu", "cpu"));
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
    .AddOutput("frames_gpu", "gpu")
    .AddOutput("labels_gpu", "gpu"));
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
    .AddOutput("frames_old", "gpu")
    .AddOutput("labels_old", "gpu"));

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
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    auto &cpu_frame_output = ws.template Output<dali::CPUBackend>(0);
    auto &gpu_frame_output = ws.template Output<dali::GPUBackend>(1);
    auto &old_frame_output = ws.template Output<dali::GPUBackend>(2);

    auto &cpu_label_output = ws.template Output<dali::CPUBackend>(3);
    auto &gpu_label_output = ws.template Output<dali::GPUBackend>(4);
    auto &old_label_output = ws.template Output<dali::GPUBackend>(5);

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

      for (int i = 0; i < sequence_length; ++i) {
        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          gpu_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        dali::CompareFrameAvgError(
          cpu_sample + i * frame_size,
          frame_buffer.data(),
          frame_size);

        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          old_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        dali::CompareFrameAvgError(
          cpu_sample + i * frame_size,
          frame_buffer.data(),
          frame_size);
      }
    }
  }
}
}  // namespace dali
