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
#include "dali/operators/video/video_utils.h"
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

TEST(VideoUtilsSecondsToTimestampTest, RoundUpAvoidsExactMatchOnFrameBoundary) {
  // Reproduces the exact case found by the video reader parity harness: a 24 fps stream with
  // timebase 1/12288 (512 ticks/frame), where file_list entry
  // "sintel_trailer_vp9_0.mp4 4 0.03536328934625863 0.9167271357878531" has its `end` timestamp
  // fall a fraction of a tick above frame 22's exact pts (22 * 512 = 11264).
  // 0.9167271357878531 s * 12288 ticks/s == 11264.743..., i.e. closer to 11265 than 11264.
  // Truncating (the old `static_cast<int64_t>` behavior) yields 11264 -- exactly frame 22's own
  // pts -- which makes GetFrameIdxByTimestamp's exact-match branch return frame 22 itself as the
  // (exclusive) end_frame, silently excluding frame 22 from the range. Legacy readers.video
  // includes frame 22. Rounding up (matching legacy's ceil(t*fps) for round-up lookups) yields
  // 11265, which is correctly recognized as being past frame 22's pts.
  AVRational timebase{1, 12288};
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.9167271357878531, /*round_down=*/false), 11265);

  // The reviewer's example: a value whose *nearest-tick* rounding (std::llround, the previous
  // fix) still lands exactly on frame 22's pts (11264), because 11264.4096 rounds down to 11264
  // -- reproducing the same exact-match bug that motivated this fix in the first place, just for
  // a narrower range of inputs. 0.91670 s * 12288 ticks/s == 11264.4096.
  //   - std::llround(11264.4096) == 11264  (bug: exact match with frame 22's own pts)
  //   - ceil(11264.4096) == 11265          (correct: unambiguously past frame 22)
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.91670, /*round_down=*/false), 11265);
  EXPECT_NE(std::llround(0.91670 * timebase.den / timebase.num), 11265);
}

TEST(VideoUtilsSecondsToTimestampTest, RoundDownMatchesLegacyFloor) {
  AVRational timebase{1, 12288};
  // Mirrors legacy's floor(t*fps) for round-down lookups (`should_round_down_start`/`_end`).
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.9167271357878531, /*round_down=*/true), 11264);
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.91670, /*round_down=*/true), 11264);
}

TEST(VideoUtilsSecondsToTimestampTest, RoundsUpOrDownAtTickBoundary) {
  AVRational timebase{1, 1000};
  // Exact tick boundaries round the same way regardless of direction.
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.002, /*round_down=*/false), 2);
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.002, /*round_down=*/true), 2);
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0, /*round_down=*/false), 0);
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0, /*round_down=*/true), 0);
  // Fractional ticks: round up goes to the next tick, round down stays at the current one.
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0015, /*round_down=*/false), 2);  // 1.5 ticks -> 2
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0015, /*round_down=*/true), 1);   // 1.5 ticks -> 1
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0014, /*round_down=*/false), 2);  // 1.4 ticks -> 2
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.0014, /*round_down=*/true), 1);   // 1.4 ticks -> 1
}

TEST(FramesDecoderConstantFrameTest, FloatNormalizedScalesFillValue) {
  Tensor<CPUBackend> frame;
  std::vector<uint8_t> fill = {0, 51, 255};
  TensorShape<> shape{2, 2, 3};
  auto *raw = ConstantFrame(frame, shape, make_cspan(fill), 0, true, DALI_FLOAT, true);
  ASSERT_EQ(frame.type(), DALI_FLOAT);
  auto *data = reinterpret_cast<const float *>(raw);
  for (int64_t i = 0; i < shape.num_elements(); i++)
    EXPECT_FLOAT_EQ(data[i], fill[i % 3] / 255.0f) << "at element " << i;
}

TEST(FramesDecoderConstantFrameTest, FloatUnnormalizedKeepsByteScale) {
  Tensor<CPUBackend> frame;
  std::vector<uint8_t> fill = {10};
  TensorShape<> shape{2, 2, 3};
  auto *data = reinterpret_cast<const float *>(
      ConstantFrame(frame, shape, make_cspan(fill), 0, true, DALI_FLOAT, false));
  for (int64_t i = 0; i < shape.num_elements(); i++)
    EXPECT_FLOAT_EQ(data[i], 10.0f) << "at element " << i;
}

TEST(FramesDecoderConstantFrameTest, ReusedBufferIsRebuiltForADifferentType) {
  Tensor<CPUBackend> frame;
  std::vector<uint8_t> fill = {255};
  TensorShape<> shape{2, 2, 3};
  ConstantFrame(frame, shape, make_cspan(fill), 0, true);  // uint8
  ASSERT_EQ(frame.type(), DALI_UINT8);
  auto *data = reinterpret_cast<const float *>(
      ConstantFrame(frame, shape, make_cspan(fill), 0, true, DALI_FLOAT, true));
  ASSERT_EQ(frame.type(), DALI_FLOAT);
  EXPECT_FLOAT_EQ(data[shape.num_elements() - 1], 1.0f);
}

namespace {

// Appends one AVCC/ISO-framed NAL unit: a `length_size`-byte big-endian length prefix followed by
// `size` bytes, the first `header.size()` of which are the NAL unit header. The rest is filled
// with a byte pattern that can't form an Annex-B start code.
void AppendAvccNal(std::vector<uint8_t> &buf, int length_size, std::vector<uint8_t> header,
                   uint32_t size) {
  for (int i = length_size - 1; i >= 0; i--)
    buf.push_back(static_cast<uint8_t>(size >> (8 * i)));
  for (uint32_t i = 0; i < size; i++)
    buf.push_back(i < header.size() ? header[i] : 0xAB);
}

void AppendAnnexBNal(std::vector<uint8_t> &buf, std::vector<uint8_t> header, uint32_t size) {
  buf.insert(buf.end(), {0, 0, 0, 1});
  for (uint32_t i = 0; i < size; i++)
    buf.push_back(i < header.size() ? header[i] : 0xAB);
}

int CountKeyframes(const FrameIndex &index) {
  int n = 0;
  for (auto &e : index.index)
    n += e.is_keyframe;
  return n;
}

// Leading bytes of real extradata, dumped from DALI_extra files.
// db/video/containers/mov/cfr.mov: AVCDecoderConfigurationRecord, lengthSizeMinusOne = 3
const std::vector<uint8_t> kAvcC = {0x01, 0x64, 0x00, 0x20, 0xff, 0xe1, 0x00, 0x19,
                                    0x67, 0x64, 0x00, 0x20, 0xac, 0xd9, 0x40, 0x50};
// db/video/containers/avi/cfr.avi: Annex-B SPS
const std::vector<uint8_t> kAnnexBExtradata = {0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x20, 0xac,
                                               0xd9, 0x40, 0x50, 0x05, 0xbb, 0x01, 0x10, 0x00};
// db/video/hevc/sintel_trailer-720p.mp4: HEVCDecoderConfigurationRecord, lengthSizeMinusOne = 3
const std::vector<uint8_t> kHvcC = {0x01, 0x01, 0x60, 0x00, 0x00, 0x00, 0x90, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x5d, 0xf0, 0x00, 0xfc,
                                    0xfd, 0xf8, 0xf8, 0x00, 0x00, 0x0f, 0x04, 0x20};

}  // namespace

TEST(NalFramingTest, LengthSizeFromExtradata) {
  using detail::GetNalLengthSize;
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_H264, kAvcC.data(), kAvcC.size()), 4);
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_HEVC, kHvcC.data(), kHvcC.size()), 4);
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_H264, kAnnexBExtradata.data(), kAnnexBExtradata.size()),
            0);
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_H264, nullptr, 0), 0);
  std::vector<uint8_t> hevc_annexb(32, 0xAB);  // Annex-B VPS
  hevc_annexb[0] = hevc_annexb[1] = hevc_annexb[2] = 0;
  hevc_annexb[3] = 1;
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_HEVC, hevc_annexb.data(), hevc_annexb.size()), 0);
  hevc_annexb[2] = 1;  // 3-byte start code
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_HEVC, hevc_annexb.data(), hevc_annexb.size()), 0);

  auto avcc2 = kAvcC;
  avcc2[4] = 0xfd;  // lengthSizeMinusOne = 1
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_H264, avcc2.data(), avcc2.size()), 2);
  auto hvcc1 = kHvcC;
  hvcc1[21] = 0x0c;  // lengthSizeMinusOne = 0
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_HEVC, hvcc1.data(), hvcc1.size()), 1);

  // Too short to hold the lengthSizeMinusOne field: fall back to Annex-B, don't read past the end
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_H264, kAvcC.data(), 4), 0);
  EXPECT_EQ(GetNalLengthSize(AV_CODEC_ID_HEVC, kHvcC.data(), 21), 0);
}

// An AVCC length prefix in [256, 511] is "00 00 01 XX" -- byte-for-byte an Annex-B start code.
TEST(NalFramingTest, AvccLengthPrefixLookingLikeStartCode) {
  using detail::HasKeyframeNalUnit;
  for (uint32_t size : {256u, 300u, 511u}) {
    std::vector<uint8_t> idr;
    AppendAvccNal(idr, 4, {0x65}, size);  // H.264 IDR slice
    EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_H264, idr.data(), idr.size(), 4)) << size;

    std::vector<uint8_t> irap;
    AppendAvccNal(irap, 4, {0x26, 0x01}, size);  // HEVC IDR_W_RADL
    EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_HEVC, irap.data(), irap.size(), 4)) << size;
  }

  // Length 0x105: the low length byte (0x05) reads as an H.264 IDR NAL header if the prefix is
  // mistaken for a start code, but the NAL unit is actually a non-IDR slice.
  std::vector<uint8_t> non_idr;
  AppendAvccNal(non_idr, 4, {0x41}, 0x105);
  EXPECT_FALSE(HasKeyframeNalUnit(AV_CODEC_ID_H264, non_idr.data(), non_idr.size(), 4));

  // A 1-byte NAL unit's prefix (00 00 00 01) is a 4-byte start code; the IDR slice follows it.
  std::vector<uint8_t> multi;
  AppendAvccNal(multi, 4, {0x09}, 1);  // access unit delimiter (header only)
  AppendAvccNal(multi, 4, {0x06, 0x05}, 300);  // SEI
  AppendAvccNal(multi, 4, {0x65}, 1000);  // IDR slice
  EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_H264, multi.data(), multi.size(), 4));
}

TEST(NalFramingTest, ShortLengthPrefixes) {
  using detail::HasKeyframeNalUnit;
  for (int length_size : {1, 2}) {
    std::vector<uint8_t> buf;
    AppendAvccNal(buf, length_size, {0x06, 0x05}, 20);  // SEI
    AppendAvccNal(buf, length_size, {0x65}, 200);  // IDR slice
    EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_H264, buf.data(), buf.size(), length_size));
    std::vector<uint8_t> p_slice;
    AppendAvccNal(p_slice, length_size, {0x41}, 200);
    EXPECT_FALSE(
        HasKeyframeNalUnit(AV_CODEC_ID_H264, p_slice.data(), p_slice.size(), length_size));
  }
}

TEST(NalFramingTest, AnnexB) {
  using detail::HasKeyframeNalUnit;
  std::vector<uint8_t> buf;
  AppendAnnexBNal(buf, {0x67, 0x64}, 20);  // SPS
  AppendAnnexBNal(buf, {0x68}, 5);  // PPS
  AppendAnnexBNal(buf, {0x65}, 300);  // IDR slice
  EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_H264, buf.data(), buf.size(), 0));
  std::vector<uint8_t> p_slice;
  AppendAnnexBNal(p_slice, {0x41}, 300);
  EXPECT_FALSE(HasKeyframeNalUnit(AV_CODEC_ID_H264, p_slice.data(), p_slice.size(), 0));
  std::vector<uint8_t> hevc;
  AppendAnnexBNal(hevc, {0x40, 0x01}, 20);  // VPS
  AppendAnnexBNal(hevc, {0x2a, 0x01}, 300);  // CRA
  EXPECT_TRUE(HasKeyframeNalUnit(AV_CODEC_ID_HEVC, hevc.data(), hevc.size(), 0));
}

TEST(NalFramingTest, TruncatedAvccPacket) {
  using detail::HasKeyframeNalUnit;
  std::vector<uint8_t> buf;
  AppendAvccNal(buf, 4, {0x65}, 300);
  // The NAL unit's declared length runs past the end of the packet: stop, don't over-read.
  EXPECT_FALSE(HasKeyframeNalUnit(AV_CODEC_ID_H264, buf.data(), buf.size() - 1, 4));
  EXPECT_FALSE(HasKeyframeNalUnit(AV_CODEC_ID_H264, buf.data(), 3, 4));
}

// Real MP4 files whose IDR slices have AVCC length prefixes in [256, 511].
TEST_F(FramesDecoderGpuTest, AvccKeyframesWithStartCodeLikeLengthPrefix) {
  FramesDecoderGpu decoder(testing::dali_extra_path() +
                           "/db/video/frame_num_timestamp/test_25fps.mp4");
  decoder.BuildIndex();
  EXPECT_EQ(CountKeyframes(decoder.GetIndex()), 1);
}

// AVI carries H.264 as Annex-B, with Annex-B extradata.
TEST_F(FramesDecoderGpuTest, AnnexBKeyframesInAvi) {
  FramesDecoderGpu decoder(testing::dali_extra_path() + "/db/video/containers/avi/cfr.avi");
  decoder.BuildIndex();
  EXPECT_EQ(CountKeyframes(decoder.GetIndex()), 1);
}

TEST_F(FramesDecoderGpuTest, AvccHevcKeyframesWithStartCodeLikeLengthPrefix) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(testing::dali_extra_path() + "/db/video/hevc/sintel_trailer-720p.mp4");
  decoder.BuildIndex();
  EXPECT_EQ(CountKeyframes(decoder.GetIndex()), 255);
}

}  // namespace dali
