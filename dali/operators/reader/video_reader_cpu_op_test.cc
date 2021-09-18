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

#include "dali/operators/reader/loader/video/video_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/cv_mat_utils.h"


namespace dali {
class VideoReaderCpuTest : public VideoTest {
};


TEST_F(VideoReaderCpuTest, CpuConstantFrameRate) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  const int step = 10;
  
  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddOutput("frames", "cpu"));

  pipe.Build({{"frames", "cpu"}});

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

    auto &frame_video_output = ws.template OutputRef<dali::CPUBackend>(0);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto sample = frame_video_output.mutable_tensor<uint8_t>(sample_id);

      for (int i = 0; i < sequence_length; ++i) {
        this->ComapreFrames(
          sample + i * this->FrameSize(video_idx), this->cfr_frames_[video_idx][gt_frame_id + i * stride].data, this->FrameSize(video_idx));

        // this->SaveFrame(sample + i * this->FrameSize(video_idx), i, sample_id, batch_id, "reader", this->Width(video_idx), this->Height(video_idx), this->Channels());
        // this->SaveFrame(this->cfr_frames_[video_idx][gt_frame_id + i * stride].data, i, sample_id, batch_id, "gt", this->Width(video_idx), this->Height(video_idx), this->Channels());
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

TEST_F(VideoReaderCpuTest, CpuVariableFrameRate) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  const int step = 10;
  
  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/vfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/vfr/test_2.mp4"})
    .AddOutput("frames", "cpu"));

  pipe.Build({{"frames", "cpu"}});

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

    auto &frame_video_output = ws.template OutputRef<dali::CPUBackend>(0);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto sample = frame_video_output.mutable_tensor<uint8_t>(sample_id);

      for (int i = 0; i < sequence_length; ++i) {
        this->ComapreFrames(
          sample + i * this->FrameSize(video_idx), this->cfr_frames_[video_idx][gt_frame_id + i * stride].data, this->FrameSize(video_idx));

        // this->SaveFrame(sample + i * this->FrameSize(video_idx), i, sample_id, batch_id, "reader", this->Width(video_idx), this->Height(video_idx), this->Channels());
        // this->SaveFrame(this->gt_frames_[video_idx][gt_frame_id + i * stride].data, i, sample_id, batch_id, "gt", this->Width(video_idx), this->Height(video_idx), this->Channels());
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

TEST_F(VideoReaderCpuTest, BenchamrkIndex) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  
  
  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddOutput("frames", "cpu"));

  pipe.Build({{"frames", "cpu"}});
}

TEST_F(VideoReaderCpuTest, CompareReaders) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  const int step = 10;
  
  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddOutput("frames", "cpu"));
  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "gpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
        testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"})
    .AddOutput("frames_gpu", "gpu"));

  pipe.Build({{"frames", "cpu"}, {"frames_gpu", "gpu"}});

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

    auto &frame_video_output = ws.template OutputRef<dali::CPUBackend>(0);
    auto &frame_gpu_video_output = ws.template OutputRef<dali::GPUBackend>(1);

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto sample = frame_video_output.mutable_tensor<uint8_t>(sample_id);
      auto sample_gpu = frame_gpu_video_output.mutable_tensor<uint8_t>(sample_id);

      vector<uint8_t> frame_gpu(this->FrameSize(video_idx));
      
      for (int i = 0; i < sequence_length; ++i) {
        MemCopy(
          frame_gpu.data(),
          sample_gpu + i * this->FrameSize(video_idx),
          FrameSize(video_idx) * sizeof(uint8_t));

        this->ComapreFrames(
          sample + i * this->FrameSize(video_idx), frame_gpu.data(), this->FrameSize(video_idx), 100);

        // this->SaveFrame(sample + i * this->FrameSize(video_idx), i, sample_id, batch_id, "reader", this->Width(video_idx), this->Height(video_idx), this->Channels());
        // this->SaveFrame(frame_gpu.data(), i, sample_id, batch_id, "gt", this->Width(video_idx), this->Height(video_idx), this->Channels());
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
