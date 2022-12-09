// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <exception>
#include <random>

#include "dali/core/cuda_error.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/test/dali_test_config.h"

#include "dali/pipeline/pipeline.h"

namespace dali {
class FramesDecoderTestBase : public VideoTestBase {
 public:
  virtual void RunSequentialForwardTest(
    FramesDecoder &decoder, TestVideo &ground_truth, double eps = 1.0) {
    // Iterate through the whole video in order
    for (int i = 0; i < decoder.NumFrames(); ++i) {
      ASSERT_EQ(decoder.NextFrameIdx(), i);
      decoder.ReadNextFrame(FrameData());
      AssertFrame(FrameData(), i, ground_truth, eps);
    }

    ASSERT_EQ(decoder.NextFrameIdx(), -1);
  }

  virtual void RunSequentialTest(
    FramesDecoder &decoder, TestVideo &ground_truth, double eps = 1.0) {
    // Iterate through the whole video in order
    RunSequentialForwardTest(decoder, ground_truth, eps);

    decoder.Reset();

    RunSequentialForwardTest(decoder, ground_truth, eps);
  }

  virtual void RunTest(FramesDecoder &decoder, TestVideo &ground_truth, double eps = 1.0) {
    ASSERT_EQ(decoder.Height(), ground_truth.Height());
    ASSERT_EQ(decoder.Width(), ground_truth.Width());
    ASSERT_EQ(decoder.Channels(), ground_truth.NumChannels());
    ASSERT_EQ(decoder.NumFrames(), ground_truth.NumFrames());
    ASSERT_EQ(decoder.IsVfr(), ground_truth.IsVfr());

    RunSequentialTest(decoder, ground_truth, eps);
    decoder.Reset();

    // Read first frame
    ASSERT_EQ(decoder.NextFrameIdx(), 0);
    decoder.ReadNextFrame(FrameData());
    AssertFrame(FrameData(), 0, ground_truth, eps);

    // Seek to frame
    decoder.SeekFrame(25);
    ASSERT_EQ(decoder.NextFrameIdx(), 25);
    decoder.ReadNextFrame(FrameData());
    AssertFrame(FrameData(), 25, ground_truth, eps);

    // Seek back to frame
    decoder.SeekFrame(12);
    ASSERT_EQ(decoder.NextFrameIdx(), 12);
    decoder.ReadNextFrame(FrameData());
    AssertFrame(FrameData(), 12, ground_truth, eps);

    // Seek to last frame (flush frame)
    int last_frame_index = ground_truth.NumFrames() - 1;
    decoder.SeekFrame(last_frame_index);
    ASSERT_EQ(decoder.NextFrameIdx(), last_frame_index);
    decoder.ReadNextFrame(FrameData());
    AssertFrame(FrameData(), last_frame_index, ground_truth, eps);
    ASSERT_EQ(decoder.NextFrameIdx(), -1);

    // Wrap around to first frame
    ASSERT_FALSE(decoder.ReadNextFrame(FrameData()));
    decoder.Reset();
    ASSERT_EQ(decoder.NextFrameIdx(), 0);
    decoder.ReadNextFrame(FrameData());
    AssertFrame(FrameData(), 0, ground_truth, eps);

    // Seek to random frames and read them
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, last_frame_index);

    for (int i = 0; i < 20; ++i) {
      int next_index = distr(gen);

      decoder.SeekFrame(next_index);
      decoder.ReadNextFrame(FrameData());
      AssertFrame(FrameData(), next_index, ground_truth, eps);
    }
  }

  virtual void AssertFrame(uint8_t *frame, int index, TestVideo& ground_truth,
                           double eps = 1.0) = 0;

  virtual uint8_t *FrameData() = 0;
};

class FramesDecoderTest_CpuOnlyTests : public FramesDecoderTestBase {
 public:
  // due to difference in CPU postprocessing on different CPUs eps is 10
  void RunSequentialTest(FramesDecoder &decoder, TestVideo &ground_truth, double eps = 10.) {
    FramesDecoderTestBase::RunSequentialTest(decoder, ground_truth, eps);
  }

  // due to difference in CPU postprocessing on different CPUs eps is 10
  void RunTest(FramesDecoder &decoder, TestVideo &ground_truth, double eps = 10.0) {
    FramesDecoderTestBase::RunTest(decoder, ground_truth, eps);
  }

  void AssertFrame(uint8_t *frame, int index, TestVideo& ground_truth, double eps = 1.0) override {
    ground_truth.CompareFrame(index, frame, eps);
  }

  void SetUp() override {
    frame_buffer_.resize(VideoTestBase::MaxFrameSize());
  }

  uint8_t *FrameData() override {
    return frame_buffer_.data();
  }

  void RunConstructorFailureTest(std::string path, std::string expected_error) {
    RunFailureTest([&]() -> void {
      FramesDecoder decoder(path);},
      expected_error);
  }

 private:
  std::vector<uint8_t> frame_buffer_;
};

class FramesDecoderGpuTest : public FramesDecoderTestBase {
 public:
  static void SetUpTestSuite() {
    VideoTestBase::SetUpTestSuite();
    DeviceGuard(0);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void RunSequentialTest(FramesDecoder &decoder, TestVideo &ground_truth, double eps = 1.5) {
    FramesDecoderTestBase::RunSequentialTest(decoder, ground_truth, eps);
  }

  void RunTest(FramesDecoder &decoder, TestVideo &ground_truth, double eps = 1.5) {
    FramesDecoderTestBase::RunTest(decoder, ground_truth, eps);
  }

  void AssertFrame(uint8_t *frame, int index, TestVideo& ground_truth, double eps = 1.0) override {
    MemCopy(FrameDataCpu(), frame, ground_truth.FrameSize());
    ground_truth.CompareFrameAvgError(index, frame_cpu_buffer_.data(), eps);
  }

  void SetUp() override {
    frame_cpu_buffer_.resize(VideoTestBase::MaxFrameSize());
    frame_gpu_buffer_.resize(VideoTestBase::MaxFrameSize());
  }

  uint8_t *FrameData() override {
    return frame_gpu_buffer_.data();
  }

  uint8_t *FrameDataCpu() {
    return frame_cpu_buffer_.data();
  }

 private:
  std::vector<uint8_t> frame_cpu_buffer_;
  DeviceBuffer<uint8_t> frame_gpu_buffer_;
};


TEST_F(FramesDecoderTest_CpuOnlyTests, ConstantFrameRate) {
  FramesDecoder decoder(cfr_videos_paths_[0]);
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, ConstantFrameRateHevc) {
  FramesDecoder decoder(cfr_hevc_videos_paths_[0]);
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRate) {
  FramesDecoder decoder(vfr_videos_paths_[1]);
  RunTest(decoder, vfr_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRateHevc) {
  FramesDecoder decoder(vfr_hevc_videos_paths_[0]);
  RunTest(decoder, vfr_hevc_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidPath) {
  std::string path = "invalid_path.mp4";

  RunConstructorFailureTest(
    path,
    make_string("Failed to open video file ", path));
}

TEST_F(FramesDecoderTest_CpuOnlyTests, NoVideoStream) {
  std::string path = testing::dali_extra_path() + "/db/audio/wav/dziendobry.wav";

  RunConstructorFailureTest(
    path,
    make_string("Could not find a valid video stream in a file ", path));
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidSeek) {
  FramesDecoder decoder(cfr_videos_paths_[0]);

  RunFailureTest([&]() -> void {
    decoder.SeekFrame(60);},
    "Invalid seek frame id. frame_id = 60, num_frames = 50");
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidCodec) {
  std::string path = testing::dali_extra_path() + "/db/video/vp9/vp9_0.mp4";

  RunConstructorFailureTest(
    path,
    make_string(
      "Unsupported video codec: vp9 in file: ", path,
      " Supported codecs: h264, HEVC."));
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRate) {
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRate) {
  FramesDecoderGpu decoder(vfr_videos_paths_[1]);
  RunTest(decoder, vfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }

  FramesDecoderGpu decoder(cfr_hevc_videos_paths_[0]);
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }

  FramesDecoderGpu decoder(vfr_hevc_videos_paths_[1]);
  RunTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InMemoryCfrVideo) {
  auto memory_video = MemoryVideo(cfr_videos_paths_[1]);

  FramesDecoder decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, InMemoryCfrVideo) {
  auto memory_video = MemoryVideo(cfr_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InMemoryVfrVideo) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[1]);

  FramesDecoder decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, InMemoryVfrVideo) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InMemoryVfrHevcVideo) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);

  FramesDecoder decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, InMemoryVfrHevcVideo) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  auto memory_video = MemoryVideo(vfr_hevc_videos_paths_[1]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRateNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);

  FramesDecoder decoder(memory_video.data(), memory_video.size(), false);
  RunSequentialTest(decoder, vfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRateHevcNoIndex) {
  auto memory_video = MemoryVideo(vfr_hevc_videos_paths_[1]);

  FramesDecoder decoder(memory_video.data(), memory_video.size(), false);
  RunSequentialTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, NoIndexSeek) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);

  FramesDecoder decoder(memory_video.data(), memory_video.size(), false);

  RunFailureTest([&]() -> void {
    decoder.SeekFrame(10);},
    "Functionality is unavailible when index is not built.");
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, vfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateHevcNoIndex) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  auto memory_video = MemoryVideo(vfr_hevc_videos_paths_[1]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4NoIndex) {
  auto memory_video = MemoryVideo(cfr_mpeg4_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, cfr_videos_[0], 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4NoIndex) {
  auto memory_video = MemoryVideo(vfr_mpeg4_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, vfr_videos_[0], 3.0);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4MkvNoIndex) {
  auto memory_video = MemoryVideo(cfr_mpeg4_mkv_videos_paths_[0]);

  FramesDecoderGpu decoder(
    memory_video.data(), memory_video.size(), 0, false, cfr_videos_[0].NumFrames());
  RunSequentialTest(decoder, cfr_videos_[0], 3.0);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4MkvNoIndexNoFrameNum) {
  auto memory_video = MemoryVideo(cfr_mpeg4_mkv_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, cfr_videos_[0], 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4MkvNoIndex) {
  auto memory_video = MemoryVideo(vfr_mpeg4_mkv_videos_paths_[1]);

  FramesDecoderGpu decoder(
    memory_video.data(), memory_video.size(), 0, false, vfr_videos_[1].NumFrames());
  RunSequentialTest(decoder, vfr_videos_[1], 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4MkvNoIndexNoFrameNum) {
  auto memory_video = MemoryVideo(vfr_mpeg4_mkv_videos_paths_[1]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialTest(decoder, vfr_videos_[1], 3.0);
}

TEST_F(FramesDecoderGpuTest, RawH264) {
  auto memory_video = MemoryVideo(cfr_raw_h264_videos_paths_[1]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialForwardTest(decoder, cfr_videos_[1], 1.5);
}

TEST_F(FramesDecoderGpuTest, RawH265) {
  auto memory_video = MemoryVideo(cfr_raw_h264_videos_paths_[0]);

  FramesDecoderGpu decoder(memory_video.data(), memory_video.size(), 0, false);
  RunSequentialForwardTest(decoder, cfr_videos_[0], 1.5);
}

}  // namespace dali
