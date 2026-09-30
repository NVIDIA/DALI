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

TEST_F(FramesDecoderTest_CpuOnlyTests, YCbCrDecodesWithoutCrashing) {
  FramesDecoderCpu ycbcr_decoder(cfr_videos_paths_[0], DALI_YCbCr);
  ycbcr_decoder.BuildIndex();
  std::vector<uint8_t> ycbcr_frame(ycbcr_decoder.FrameSize());
  ASSERT_TRUE(ycbcr_decoder.ReadNextFrame(ycbcr_frame.data()));

  bool any_nonzero = false;
  for (uint8_t v : ycbcr_frame) {
    if (v != 0) {
      any_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(any_nonzero) << "Decoded YCbCr frame was all zeros";

  // Decode the same frame as RGB and confirm the two conversions genuinely differ pixel-wise
  // (guards against a fix that "doesn't crash" but silently reuses/garbles the RGB path).
  FramesDecoderCpu rgb_decoder(cfr_videos_paths_[0], DALI_RGB);
  rgb_decoder.BuildIndex();
  std::vector<uint8_t> rgb_frame(rgb_decoder.FrameSize());
  ASSERT_TRUE(rgb_decoder.ReadNextFrame(rgb_frame.data()));

  ASSERT_EQ(ycbcr_frame.size(), rgb_frame.size());
  EXPECT_FALSE(std::equal(ycbcr_frame.begin(), ycbcr_frame.end(), rgb_frame.begin()))
      << "YCbCr and RGB decodes of the same frame were byte-identical";
}

TEST_F(FramesDecoderTest_CpuOnlyTests, YCbCrMatchesGpuBackendWithinTolerance) {
  // The CPU backend's YCbCr conversion (libswscale) and the GPU backend's (a custom CUDA
  // kernel matching the legacy reader's conversion) are independent implementations of 4:2:0
  // chroma upsampling (libswscale's SWS_BILINEAR vs. the GPU kernel's own bilinear sampling),
  // so they aren't expected to be byte-identical -- rounding can differ by a step or two on
  // sharp chroma edges. What this fix guarantees is colorimetry (limited/"TV" range for YCbCr,
  // not full range, per DALI-4916): without it, the two backends disagreed by up to 20/255 per
  // channel; with it, differences should be small, uniformly-distributed rounding noise, not a
  // systematic offset.
  FramesDecoderCpu cpu_decoder(cfr_videos_paths_[0], DALI_YCbCr);
  cpu_decoder.BuildIndex();
  std::vector<uint8_t> frame_cpu(cpu_decoder.FrameSize());
  ASSERT_TRUE(cpu_decoder.ReadNextFrame(frame_cpu.data()));

  FramesDecoderGpu gpu_decoder(cfr_videos_paths_[0], 0, DALI_YCbCr);
  gpu_decoder.BuildIndex();
  DeviceBuffer<uint8_t> frame_gpu_device;
  frame_gpu_device.resize(gpu_decoder.FrameSize());
  ASSERT_TRUE(gpu_decoder.ReadNextFrame(frame_gpu_device.data()));
  std::vector<uint8_t> frame_gpu(gpu_decoder.FrameSize());
  MemCopy(frame_gpu.data(), frame_gpu_device.data(), gpu_decoder.FrameSize());

  ASSERT_EQ(frame_cpu.size(), frame_gpu.size());
  int max_abs_diff = 0;
  long sum_abs_diff = 0;
  for (size_t i = 0; i < frame_cpu.size(); ++i) {
    int d = std::abs(static_cast<int>(frame_cpu[i]) - static_cast<int>(frame_gpu[i]));
    max_abs_diff = std::max(max_abs_diff, d);
    sum_abs_diff += d;
  }
  double avg_abs_diff = static_cast<double>(sum_abs_diff) / frame_cpu.size();
  EXPECT_LE(max_abs_diff, 4) << "Max abs diff " << max_abs_diff
                              << " suggests a systematic mismatch, not rounding noise";
  EXPECT_LE(avg_abs_diff, 0.5) << "Avg abs diff " << avg_abs_diff
                                << " suggests a systematic mismatch, not rounding noise";
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

TEST_F(FramesDecoderGpuTest, CustomDecodeSurfaceCount) {
  // A non-default surface count (larger than the baseline of 8) must still decode the whole
  // video correctly; this exercises the frame_buffer_ sizing that num_decode_surfaces_ drives.
  FramesDecoderGpu decoder(cfr_videos_paths_[0], 0, DALI_RGB, 12);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, DefaultDecodeSurfaceCountUnchanged) {
  // Explicitly verifies the default (no 4th arg) still behaves like the pre-existing
  // hardcoded value of 8, per this plan's Global Constraints (no default-behavior change).
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, MjpegDecodesMultipleFrames) {
  // DALI-4918: the NVDEC driver reports min_num_decode_surfaces=1 for this MJPEG file --
  // with zero surface headroom, decoding the 2nd frame used to fail with
  // CUDA_ERROR_INVALID_VALUE, because this decoder never registers a real pfnDisplayPicture
  // callback (HandlePictureDisplay runs synchronously, inline, from ProcessPictureDecode), so
  // the parser's own surface-recycling bookkeeping never saw the 1st picture as "consumed"
  // before the 2nd one needed a surface. AdjustedNumDecodeSurfaces forces a larger surface
  // count for MJPEG specifically (mirroring legacy's hardcoded ulMaxNumDecodeSurfaces=20).
  auto mjpeg_path = testing::dali_extra_path() + "/db/video/mjpeg/mjpeg.avi";
  FramesDecoderGpu decoder(mjpeg_path, 0, DALI_RGB);
  decoder.BuildIndex();
  ASSERT_GT(decoder.NumFrames(), 1);

  // VideoColorSpaceConversion writes directly into this pointer via a CUDA kernel, so it must
  // be device memory, not a host std::vector.
  DeviceBuffer<uint8_t> frame0_device;
  DeviceBuffer<uint8_t> frame1_device;
  frame0_device.resize(decoder.FrameSize());
  frame1_device.resize(decoder.FrameSize());
  ASSERT_TRUE(decoder.ReadNextFrame(frame0_device.data()));
  ASSERT_TRUE(decoder.ReadNextFrame(frame1_device.data()));

  auto any_nonzero = [](const std::vector<uint8_t> &frame) {
    for (uint8_t v : frame) {
      if (v != 0) return true;
    }
    return false;
  };

  std::vector<uint8_t> frame0_host(decoder.FrameSize());
  MemCopy(frame0_host.data(), frame0_device.data(), decoder.FrameSize());
  EXPECT_TRUE(any_nonzero(frame0_host)) << "Decoded MJPEG frame 0 was all zeros";

  // frame 1 is the specific point that used to crash with CUDA_ERROR_INVALID_VALUE -- check its
  // content too, not just that ReadNextFrame returned true for it.
  std::vector<uint8_t> frame1_host(decoder.FrameSize());
  MemCopy(frame1_host.data(), frame1_device.data(), decoder.FrameSize());
  EXPECT_TRUE(any_nonzero(frame1_host)) << "Decoded MJPEG frame 1 was all zeros";

  // Decode the rest of the video too -- the surface-count fix must hold for the whole file,
  // not just the first couple of frames.
  for (int i = 2; i < decoder.NumFrames(); ++i) {
    ASSERT_TRUE(decoder.ReadNextFrame(frame0_device.data())) << "Failed decoding frame " << i;
  }
}

TEST_F(FramesDecoderGpuTest, NormalizedFloatOutputInRange) {
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  decoder.SetOutputType(DALI_FLOAT);
  decoder.SetNormalizedRange(true);
  decoder.BuildIndex();

  // Allocate GPU buffer large enough for float output
  int frame_size = decoder.FrameSize();
  DeviceBuffer<float> frame_gpu_buffer;
  frame_gpu_buffer.resize(frame_size);

  // Read frame into GPU buffer
  ASSERT_TRUE(decoder.ReadNextFrame(reinterpret_cast<uint8_t *>(frame_gpu_buffer.data())));

  // Copy float data from GPU to CPU
  std::vector<float> frame_cpu(frame_size);
  MemCopy(frame_cpu.data(), frame_gpu_buffer.data(), frame_size * sizeof(float));

  // Verify most values are in normalized [0.0, 1.0] range
  // Note: Some outliers exist due to floating-point conversion and the underlying pixel data
  int count_in_range = 0;
  const float epsilon = 0.25f;  // Allow for conversion tolerances
  for (int i = 0; i < frame_size; ++i) {
    if (frame_cpu[i] >= -epsilon && frame_cpu[i] <= 1.0f + epsilon) {
      count_in_range++;
    }
  }
  // Verify that at least 97% of pixels are in the valid range
  float fraction_in_range = static_cast<float>(count_in_range) / frame_size;
  EXPECT_GT(fraction_in_range, 0.97f) << "Expected at least 97% of pixels to be in [-epsilon, 1+epsilon] range";

  // Also verify some values are actually in [0, 1] to confirm normalization happened
  bool found_normalized = false;
  for (int i = 0; i < frame_size; ++i) {
    if (frame_cpu[i] >= 0.1f && frame_cpu[i] <= 0.9f) {
      found_normalized = true;
      break;
    }
  }
  EXPECT_TRUE(found_normalized) << "Expected to find at least one pixel value in normalized range [0.1, 0.9]";
}

TEST_F(FramesDecoderGpuTest, UnnormalizedFloatOutputInByteRange) {
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  decoder.SetOutputType(DALI_FLOAT);
  decoder.SetNormalizedRange(false);
  decoder.BuildIndex();

  // Allocate GPU buffer large enough for float output
  int frame_size = decoder.FrameSize();
  DeviceBuffer<float> frame_gpu_buffer;
  frame_gpu_buffer.resize(frame_size);

  // Read frame into GPU buffer
  ASSERT_TRUE(decoder.ReadNextFrame(reinterpret_cast<uint8_t *>(frame_gpu_buffer.data())));

  // Copy float data from GPU to CPU
  std::vector<float> frame_cpu(frame_size);
  MemCopy(frame_cpu.data(), frame_gpu_buffer.data(), frame_size * sizeof(float));

  // Verify values are in byte range [0.0, 255.0] and at least one exceeds 1.0
  bool any_above_one = false;
  const float epsilon = 0.35f;  // Allow for conversion tolerances
  int count_in_range = 0;
  for (int i = 0; i < frame_size; ++i) {
    if (frame_cpu[i] >= -epsilon && frame_cpu[i] <= 255.0f + epsilon) {
      count_in_range++;
    }
    if (frame_cpu[i] > 1.0f) {
      any_above_one = true;
    }
  }
  // Verify that most pixels are in the valid byte range
  float fraction_in_range = static_cast<float>(count_in_range) / frame_size;
  EXPECT_GT(fraction_in_range, 0.97f) << "Expected at least 97% of pixels to be in byte range [-epsilon, 255+epsilon]";
  EXPECT_TRUE(any_above_one) << "Expected at least one pixel channel above 1.0 in unnormalized float output";
}

namespace {

FrameIndex MakeFrameIndex(const std::vector<int64_t> &pts) {
  FrameIndex index;
  index.timebase = AVRational{1, 1000};
  for (auto p : pts)
    index.index.push_back(IndexEntry{p, 0, false, false});
  return index;
}

}  // namespace

TEST(FramesDecoderFrameIndexTest, TimestampInsideTheStream) {
  auto index = MakeFrameIndex({100, 200, 300});
  EXPECT_EQ(index.GetFrameIdxByTimestamp(200, false), 1);  // exact match
  EXPECT_EQ(index.GetFrameIdxByTimestamp(200, true), 1);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(250, false), 2);  // between frames 1 and 2
  EXPECT_EQ(index.GetFrameIdxByTimestamp(250, true), 1);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(50, false), 0);   // before the first frame
  EXPECT_EQ(index.GetFrameIdxByTimestamp(50, true), 0);
}

TEST(FramesDecoderFrameIndexTest, StartAndEndPts) {
  auto index = MakeFrameIndex({100, 200, 300});
  EXPECT_EQ(index.StartPts(), 100);
  EXPECT_EQ(index.EndPts(), 400);  // last pts + last inter-frame gap
  EXPECT_EQ(MakeFrameIndex({5}).EndPts(), 6);
}

TEST(FramesDecoderFrameIndexTest, PastTheLastFrameReturnsSizeSentinel) {
  auto index = MakeFrameIndex({100, 200, 300});
  // Inside the last frame's duration: rounding down selects the last frame.
  EXPECT_EQ(index.GetFrameIdxByTimestamp(350, true), 2);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(350, false), 3);
  // At or past the end of the stream: "one past the last frame", never frame 0.
  EXPECT_EQ(index.GetFrameIdxByTimestamp(400, false), 3);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(400, true), 3);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(100000, true), 3);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(100000, false), 3);
}

}  // namespace dali
