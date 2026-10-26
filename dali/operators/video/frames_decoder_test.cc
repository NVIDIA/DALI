// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <exception>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "dali/core/cuda_error.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/video_test.h"
#include "dali/test/dali_test_config.h"

#include "dali/pipeline/pipeline.h"

namespace dali {
class FramesDecoderTestBase : public VideoTestBase {
 public:
  virtual void RunSequentialForwardTest(
    FramesDecoderBase &decoder, TestVideo &ground_truth, double eps = 1.0) {
    // Iterate through the whole video in order
    for (int i = 0; i < decoder.NumFrames(); ++i) {
      ASSERT_EQ(decoder.NextFrameIdx(), i);
      decoder.ReadNextFrame(FrameData());
      AssertFrame(FrameData(), i, ground_truth, eps);
    }

    ASSERT_EQ(decoder.NextFrameIdx(), -1);
  }

  virtual void RunSequentialTest(FramesDecoderBase &decoder, TestVideo &ground_truth,
                                 double eps = 1.0) {
    // Iterate through the whole video in order
    RunSequentialForwardTest(decoder, ground_truth, eps);

    decoder.Reset();

    RunSequentialForwardTest(decoder, ground_truth, eps);
  }

  virtual void RunTest(FramesDecoderBase &decoder, TestVideo &ground_truth, bool has_index = true,
                       double eps = 1.0) {
    ASSERT_EQ(decoder.Height(), ground_truth.Height());
    ASSERT_EQ(decoder.Width(), ground_truth.Width());
    ASSERT_EQ(decoder.Channels(), ground_truth.NumChannels());
    ASSERT_EQ(decoder.NumFrames(), ground_truth.NumFrames());
    if (has_index) {
      ASSERT_EQ(decoder.IsVfr(), ground_truth.IsVfr());
    }

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
  void RunSequentialTest(FramesDecoderBase &decoder, TestVideo &ground_truth, double eps = 10.) {
    FramesDecoderTestBase::RunSequentialTest(decoder, ground_truth, eps);
  }

  // due to difference in CPU postprocessing on different CPUs eps is 10
  void RunTest(FramesDecoderBase &decoder, TestVideo &ground_truth, bool has_index = true,
               double eps = 10.0) {
    FramesDecoderTestBase::RunTest(decoder, ground_truth, has_index, eps);
  }

  void AssertFrame(uint8_t *frame, int index, TestVideo &ground_truth, double eps = 1.0) override {
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
      FramesDecoderCpu decoder(path);},
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

  void RunSequentialTest(FramesDecoderBase &decoder, TestVideo &ground_truth, double eps = 1.5) {
    FramesDecoderTestBase::RunSequentialTest(decoder, ground_truth, eps);
  }

  void RunTest(FramesDecoderBase &decoder, TestVideo &ground_truth, bool has_index = true,
               double eps = 1.5) {
    FramesDecoderTestBase::RunTest(decoder, ground_truth, has_index, eps);
  }

  void AssertFrame(uint8_t *frame, int index, TestVideo& ground_truth, double eps = 1.0) override {
    MemCopy(FrameDataCpu(), frame, ground_truth.FrameSize());
    CompareFrameAvgError(index, ground_truth.FrameSize(), ground_truth.Width(),
                         ground_truth.Height(), frame_cpu_buffer_.data(),
                         ground_truth.frames_[index].data, eps);
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

namespace {

std::string SintelClip(int label, int clip) {
  return make_string(testing::dali_extra_path(), "/db/video/sintel/labelled_videos/", label,
                     "/sintel_trailer-720p_", clip, ".mp4");
}

// The frames that SeekFrame seeks to: the last keyframes of the frames
std::set<int> SeekTargets(const FrameIndex &index) {
  std::set<int> targets;
  for (size_t i = 0; i < index.size(); i++)
    targets.insert(index[i].last_keyframe_id);
  return targets;
}

// Checks the decoding order in the index of a file without hidden packets, where a seek to a
// keyframe lands on the keyframe itself
void CheckDecodeOrder(const FrameIndex &index) {
  int n = index.size();
  std::vector<int> decode_idx;
  for (int i = 0; i < n; i++)
    decode_idx.push_back(index[i].decode_idx);
  std::sort(decode_idx.begin(), decode_idx.end());
  for (int i = 0; i < n; i++)
    ASSERT_EQ(decode_idx[i], i) << "not a permutation of the decoding positions [0, " << n << ")";
  auto targets = SeekTargets(index);
  for (int i = 0; i < n; i++) {
    if (index[i].is_keyframe) {
      EXPECT_TRUE(targets.count(i)) << "keyframe " << i;
    }
    if (targets.count(i)) {
      EXPECT_EQ(index[i].seek_decode_idx, index[i].decode_idx) << "seek to frame " << i;
    } else {
      EXPECT_EQ(index[i].seek_decode_idx, -1) << "frame " << i << " is not a seek target";
    }
  }
}

}  // namespace

TEST_F(FramesDecoderTest_CpuOnlyTests, IndexDecodeOrder) {
  for (auto &path : {cfr_videos_paths_[0], vfr_videos_paths_[0]}) {
    FramesDecoderCpu decoder(path);
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    CheckDecodeOrder(index);
    for (size_t i = 0; i < index.size(); i++)  // VP9 has no reordering
      EXPECT_EQ(index[i].decode_idx, static_cast<int>(i)) << path;
  }
}

TEST_F(FramesDecoderGpuTest, IndexDecodeOrder) {
  auto h264_path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";
  for (auto &path : {cfr_videos_paths_[0], h264_path}) {
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    CheckDecodeOrder(decoder.GetIndex());
  }
}

// The Sintel clips start with packets that an edit list hides (negative timestamps): they are not
// in the index, but they are decoded, before the first frame or after a seek that lands on them
TEST_F(FramesDecoderGpuTest, IndexDecodeOrderLeadIn) {
  {
    // 310 packets, of which 188 hidden, and only the IDR at frame 62 is shown
    FramesDecoderGpu decoder(SintelClip(2, 8));
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    ASSERT_EQ(index.size(), 122u);
    std::set<int> positions;
    for (size_t i = 0; i < index.size(); i++) {
      EXPECT_GE(index.DecodeIdx(i), 0);
      EXPECT_LT(index.DecodeIdx(i), 310);
      positions.insert(index.DecodeIdx(i));
    }
    EXPECT_EQ(positions.size(), index.size());  // the other 188 are hidden
    EXPECT_FALSE(index[0].is_keyframe);
    EXPECT_EQ(index[0].last_keyframe_id, 0);
    EXPECT_EQ(index.SeekDecodeIdx(0), 0);  // the hidden IDR
    int decoded = 0;
    for (int i = 0; i < 32; i++)
      decoded = std::max(decoded, index.DecodeIdx(i) + 1);
    EXPECT_EQ(decoded, 221);  // the pictures decoded for the first 32 frames
    ASSERT_TRUE(index[62].is_keyframe);
    EXPECT_EQ(index.DecodeIdx(62), 250);
    EXPECT_EQ(index.SeekDecodeIdx(62), 250);
  }
  // Only the IDR is hidden: every frame is decoded from the start
  for (auto &path : {SintelClip(0, 3), SintelClip(1, 7)}) {
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    for (size_t i = 0; i < index.size(); i++)
      EXPECT_EQ(index[i].last_keyframe_id, 0) << path << " frame " << i;
    EXPECT_EQ(index.SeekDecodeIdx(0), 0) << path;
  }
  {
    // The seek to the shown IDR (frame 60, packet 118) lands on the hidden one: mov searches the
    // sync samples by decoding timestamp, offset by the edit list
    FramesDecoderGpu decoder(SintelClip(0, 9));
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    ASSERT_TRUE(index[60].is_keyframe);
    EXPECT_EQ(index.DecodeIdx(60), 118);
    EXPECT_EQ(index.SeekDecodeIdx(60), 0);
  }
  {
    // Here the seeks to the shown IDRs land on them
    FramesDecoderGpu decoder(SintelClip(2, 5));
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    for (int k : {21, 58}) {
      ASSERT_TRUE(index[k].is_keyframe) << k;
      EXPECT_EQ(index.SeekDecodeIdx(k), index.DecodeIdx(k)) << k;
    }
    EXPECT_EQ(index.DecodeIdx(21), 37);
    EXPECT_EQ(index.DecodeIdx(58), 74);
  }
}

TEST_F(FramesDecoderTest_CpuOnlyTests, ConstantFrameRate) {
  FramesDecoderCpu decoder(cfr_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRate) {
  FramesDecoderCpu decoder(vfr_videos_paths_[1]);
  decoder.BuildIndex();
  RunTest(decoder, vfr_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidSeek) {
  FramesDecoderCpu decoder(cfr_videos_paths_[0]);
  decoder.BuildIndex();
  RunFailureTest([&]() -> void {
    decoder.SeekFrame(60);},
    "Invalid seek frame id. frame_id = 60, num_frames = 50");
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRate) {
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRate) {
  FramesDecoderGpu decoder(vfr_videos_paths_[1]);
  decoder.BuildIndex();
  RunTest(decoder, vfr_videos_[1]);
}

// H.264 is coded in whole macroblocks, so an 800x600 stream is coded as 800x608 and cropped to the
// display area. The NVDEC decoder cache has to recognize such a stream, so that a decoder is reused
// when the next file is opened and not created from scratch every time. The number of created
// decoders is checked, as the size of the cache would not change if it is full and evicts one.
TEST_F(FramesDecoderGpuTest, DecoderCacheReusedForCroppedVideo) {
  std::string path = testing::dali_extra_path() + "/db/video/cfr/test_2.mp4";
  auto decode_first_frame = [&]() {
    FramesDecoderGpu decoder(path);
    ASSERT_TRUE(decoder.IsValid());
    decoder.BuildIndex();
    ASSERT_EQ(decoder.Height(), 600);
    ASSERT_EQ(decoder.Width(), 800);
    decoder.ReadNextFrame(FrameData());
  };

  // The first decoder may or may not be created, depending on what ran before.
  decode_first_frame();
  size_t num_created = frame_dec_gpu_impl::NVDECDecodersCreated();
  // The decoder returned to the cache by the first one has to be reused.
  decode_first_frame();
  EXPECT_EQ(frame_dec_gpu_impl::NVDECDecodersCreated(), num_created);
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(cfr_hevc_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(vfr_hevc_videos_paths_[1]);
  decoder.BuildIndex();
  RunTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InMemoryCfrVideo) {
  auto memory_video = MemoryVideo(cfr_videos_paths_[1]);
  FramesDecoderCpu decoder(memory_video.data(), memory_video.size());
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, InMemoryCfrVideo) {
  auto memory_video = MemoryVideo(cfr_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InMemoryVfrVideo) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[1]);
  FramesDecoderCpu decoder(memory_video.data(), memory_video.size());
  decoder.BuildIndex();
  RunTest(decoder, vfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, InMemoryVfrVideo) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  decoder.BuildIndex();
  RunTest(decoder, vfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, InMemoryVfrHevcVideo) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  auto memory_video = MemoryVideo(vfr_hevc_videos_paths_[1]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  decoder.BuildIndex();
  RunTest(decoder, vfr_hevc_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRateNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderCpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0], false);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, NoIndexSeek) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderCpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0], false);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0], false);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateHevcNoIndex) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  auto memory_video = MemoryVideo(vfr_hevc_videos_paths_[1]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_hevc_videos_[1], false);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4NoIndex) {
  auto memory_video = MemoryVideo(cfr_mpeg4_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[0], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4NoIndex) {
  auto memory_video = MemoryVideo(vfr_mpeg4_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[0], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4MkvNoIndex) {
  auto memory_video = MemoryVideo(cfr_mpeg4_mkv_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  decoder.SetNumFrames(cfr_videos_[0].NumFrames());
  RunTest(decoder, cfr_videos_[0], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, CfrFrameRateMpeg4MkvNoIndexNoFrameNum) {
  auto memory_video = MemoryVideo(cfr_mpeg4_mkv_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[0], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4MkvNoIndex) {
  auto memory_video = MemoryVideo(vfr_mpeg4_mkv_videos_paths_[1]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  decoder.SetNumFrames(vfr_videos_[1].NumFrames());
  RunTest(decoder, vfr_videos_[1], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, VfrFrameRateMpeg4MkvNoIndexNoFrameNum) {
  auto memory_video = MemoryVideo(vfr_mpeg4_mkv_videos_paths_[1]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, vfr_videos_[1], false, 3.0);
}

TEST_F(FramesDecoderGpuTest, RawH264) {
  auto memory_video = MemoryVideo(cfr_raw_h264_videos_paths_[1]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[1], false, 1.5);
}

TEST_F(FramesDecoderGpuTest, RawH265) {
  auto memory_video = MemoryVideo(cfr_raw_h264_videos_paths_[0]);
  FramesDecoderGpu decoder(memory_video.data(), memory_video.size());
  RunTest(decoder, cfr_videos_[0], false, 1.5);
}

}  // namespace dali
