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
class VideoReaderDecoderCpuTest : public VideoTestBase {
};

class VideoReaderDecoderGpuTest : public VideoTestBase {
};

TEST_F(VideoReaderDecoderCpuTest, CpuConstantFrameRate_CpuOnlyTests) {
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
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
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
        this->CompareFrames(
          sample + i * this->FrameSize(video_idx),
          this->GetCfrFrame(video_idx, gt_frame_id + i * stride),
          this->FrameSize(video_idx));
      }

      gt_frame_id += step;
      ++sequence_id;

      if (gt_frame_id + stride * sequence_length >= this->NumFrames(video_idx)) {
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

TEST_F(VideoReaderDecoderCpuTest, CpuVariableFrameRate_CpuOnlyTests) {
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
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/vfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/vfr/test_2.mp4"})
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
        this->CompareFrames(
          sample + i * this->FrameSize(video_idx),
          this->GetVfrFrame(video_idx, gt_frame_id + i * stride),
          this->FrameSize(video_idx));
      }

      gt_frame_id += step;
      ++sequence_id;

      if (gt_frame_id + stride * sequence_length >= this->NumFrames(video_idx)) {
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

TEST_F(VideoReaderDecoderGpuTest, GpuVariableFrameRate) {
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
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/vfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/vfr/test_2.mp4"})
    .AddArg("labels", std::vector<int>{0, 1})
    .AddArg("initial_fill", 1)
    .AddArg("lazy_init", true)
    .AddOutput("frames", "gpu")
    .AddOutput("labels", "gpu"));

  pipe.Build({{"frames", "gpu"}, {"labels", "gpu"}});

  int num_sequences = 20;
  int sequence_id = 0;
  int batch_id = 0;
  int gt_frame_id = 0;

  int video_idx = 0;

  std::vector<uint8_t> frame_cpu(std::max(
    this->FrameSize(0), this->FrameSize(1)));

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

      ASSERT_TRUE(label_cpu == video_idx);

      for (int i = 0; i < sequence_length; ++i) {
        MemCopy(
          frame_cpu.data(), sample + i * this->FrameSize(video_idx),   this->FrameSize(video_idx));
        this->CompareFramesAvgError(
          frame_cpu.data(),
          this->GetVfrFrame(video_idx, gt_frame_id + i * stride),
          this->FrameSize(video_idx));
      }

      gt_frame_id += step;
      ++sequence_id;

      if (gt_frame_id + stride * sequence_length >= this->NumFrames(video_idx)) {
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

TEST_F(VideoReaderDecoderCpuTest, RandomShuffle_CpuOnlyTests) {
  const int batch_size = 1;
  const int sequence_length = 1;
  const int seed = 1;

  Pipeline pipe(batch_size, 1, dali::CPU_ONLY_DEVICE_ID, seed);

  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("random_shuffle", true)
    .AddArg("initial_fill", this->NumFrames(0))
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
    CompareFrames(sample, this->GetCfrFrame(0, expected_order[sequence_id]), this->FrameSize(0));
  }
}

TEST_F(VideoReaderDecoderCpuTest, CompareReaders) {
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
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddArg("labels", std::vector<int>{0, 1})
    .AddOutput("frames", "cpu")
    .AddOutput("labels", "cpu"));
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
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddArg("labels", std::vector<int>{0, 1})
    .AddOutput("frames_gpu", "gpu")
    .AddOutput("labels_gpu", "gpu"));

  pipe.Build({{"frames", "cpu"}, {"frames_gpu", "gpu"}, {"labels", "cpu"}, {"labels_gpu", "gpu"}});

  int num_sequences = 20;
  int sequence_id = 0;
  int batch_id = 0;
  int gt_frame_id = 0;

  int video_idx = 0;

  vector<uint8_t> frame_gpu;
  frame_gpu.reserve(std::max(this->FrameSize(0), this->FrameSize(1)));

  while (sequence_id < num_sequences) {
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    auto &frame_video_output = ws.template Output<dali::CPUBackend>(0);
    auto &frame_gpu_video_output = ws.template Output<dali::GPUBackend>(1);

    auto &frame_label_output = ws.template Output<dali::CPUBackend>(2);
    auto &frame_gpu_label_output = ws.template Output<dali::GPUBackend>(3);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      const auto sample = frame_video_output.tensor<uint8_t>(sample_id);
      const auto sample_gpu = frame_gpu_video_output.tensor<uint8_t>(sample_id);

      const auto label = frame_label_output.tensor<int>(sample_id);
      const auto label_gpu = frame_gpu_label_output.tensor<int>(sample_id);

      ASSERT_EQ(frame_video_output.tensor_shape(sample_id),
                frame_gpu_video_output.tensor_shape(sample_id));

      int label_gpu_out = -1;
      MemCopy(&label_gpu_out, label_gpu, sizeof(int));
      ASSERT_EQ(label[0], label_gpu_out);

      frame_gpu.resize(this->FrameSize(video_idx));

      for (int i = 0; i < sequence_length; ++i) {
        frame_gpu.clear();
        MemCopy(
          frame_gpu.data(),
          sample_gpu + i * this->FrameSize(video_idx),
          FrameSize(video_idx) * sizeof(uint8_t));

        this->CompareFrames(
          sample + i * this->FrameSize(video_idx),
          frame_gpu.data(),
          this->FrameSize(video_idx), 100);
      }

      gt_frame_id += step;
      ++sequence_id;

      if (gt_frame_id + stride * sequence_length >= this->NumFrames(video_idx)) {
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

}  // namespace dali
