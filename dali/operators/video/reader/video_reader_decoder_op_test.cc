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
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dali/operators/video/video_test.h"
#include "dali/operators/video/reader/video_reader_decoder_op.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/cv_mat_utils.h"
#include "dali/core/dev_buffer.h"

// This runs all the tests from the legacy video reader op in addition to those defined in this file
#define VIDEO_READER_OP experimental__readers__Video
#include "dali/operators/video/legacy/reader/video_reader_op_test.cc"

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

        // The next sequence exists only if its last frame is inside the video (same fit
        // criterion as VideoLoaderDecoder::PrepareMetadataImpl and legacy readers.video).
        if (gt_frame_id + stride * (sequence_length - 1) >=
            ground_truth_videos[video_idx].NumFrames()) {
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
      .AddArg("enable_frame_num", "scalar")
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
    CompareFrameAvgError(frame_id, ground_truth.FrameSize(), ground_truth.Width(),
                         ground_truth.Height(), frame, ground_truth.frames_[frame_id].data, 1.5);
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
    CompareFrameAvgError(frame_id, ground_truth.FrameSize(), ground_truth.Width(),
                         ground_truth.Height(), frame_buffer_.data(), ground_truth.frames_[frame_id].data, 1.5);
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
  RunShuffleTest<dali::GPUBackend>();
}

// Regression test for a decoder being cached (GetOrOpenDecoder's decoder_cache_) before it is
// validated: BuildIndex()/IsValid() must all pass *before* a decoder is inserted into
// decoder_cache_, so a failing filename never leaves a broken, unvalidated entry behind for a
// later call to silently reuse. This calls VideoReaderDecoder::GetOrOpenDecoder() directly
// (rather than through the pipeline's background prefetch thread), because an exception from
// the prefetch thread permanently stops it (DataReader::PrefetchWorker's catch block), so the
// bug can't be observed by running the same pipeline instance twice after a failure. An MPEG-PS
// file is used as the failing input: it opens fine, but BuildIndex() rejects it.
TEST_F(VideoReaderDecoderCpuTest, DecoderNotCachedBeforeValidation_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  pipe.Build({{"frames", "cpu"}});

  auto *op = dynamic_cast<VideoReaderDecoder<CPUBackend> *>(pipe.GetOperator("r"));
  ASSERT_NE(op, nullptr);

  const std::string mpeg_ps_path =
      testing::dali_extra_path() + "/db/video/containers/mpeg/cfr.mpeg";
  bool is_new_decoder = false;
  // First call: BuildIndex() rejects MPEG-PS, so this must throw -- and must not leave a cache
  // entry behind.
  EXPECT_THROW(op->GetOrOpenDecoder(mpeg_ps_path, is_new_decoder), DALIException);

  // Second call for the *same* filename: on the buggy code (cache insertion before
  // validation), this would return the already-cached, never-indexed decoder
  // (is_new_decoder=false), silently "succeeding". With the fix, nothing was cached, so this
  // must construct and validate again, and fail again.
  EXPECT_THROW(op->GetOrOpenDecoder(mpeg_ps_path, is_new_decoder), DALIException);
}

// require_constant_frame_rate=True must reject a VFR file while the dataset metadata is built
// (i.e. at pipeline build), not only once the file is first decoded mid-epoch.
TEST_F(VideoReaderDecoderCpuTest, RequireConstantFrameRateFailsAtBuild_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("require_constant_frame_rate", true)
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0], vfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  try {
    pipe.Build({{"frames", "cpu"}});
    FAIL() << "Expected pipeline build to fail for a variable frame rate video";
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find("variable frame rate"), std::string::npos) << e.what();
  }
}

TEST_F(VideoReaderDecoderCpuTest, RequireConstantFrameRateAcceptsCfr_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("require_constant_frame_rate", true)
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  EXPECT_NO_THROW(pipe.Build({{"frames", "cpu"}}));
}

namespace {

/**
 * @brief Writes `contents` to a fresh temporary file; the file is removed when the returned
 *        object goes out of scope.
 */
struct TempFileList {
  explicit TempFileList(const std::string &contents) {
    char path_template[] = "/tmp/video_reader_decoder_test_XXXXXX";
    int fd = mkstemp(path_template);
    close(fd);
    path = path_template;
    std::ofstream f(path);
    f << contents;
  }
  ~TempFileList() {
    std::remove(path.c_str());
  }
  std::string path;
};

}  // namespace

class VideoReaderDecoderFileListTest : public VideoReaderDecoderCpuTest {
 protected:
  /**
   * @brief Builds a CPU reader (sequence_length=1, file_list_format='frames') over a one-entry
   *        file list for cfr_videos_paths_[0] and returns its epoch size.
   */
  int64_t FramesModeEpochSize(const std::string &start_end) {
    TempFileList file_list(cfr_videos_paths_[0] + " 0 " + start_end + "\n");
    Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
    pipe.AddOperator(OpSpec("experimental__readers__Video")
      .AddArg("device", "cpu")
      .AddArg("sequence_length", 1)
      .AddArg("file_list", file_list.path)
      .AddArg("file_list_format", "frames")
      .AddOutput("frames", StorageDevice::CPU)
      .AddOutput("labels", StorageDevice::CPU), "r");
    pipe.Build({{"frames", "cpu"}, {"labels", "cpu"}});
    return pipe.GetReaderMeta("r").epoch_size;
  }
};

// A negative start counts back from the end and is clamped at the first frame.
TEST_F(VideoReaderDecoderFileListTest, FramesNegativeStartClampedToZero_CpuOnlyTests) {
  int num_frames = cfr_videos_[0].NumFrames();
  EXPECT_EQ(FramesModeEpochSize(std::to_string(-(num_frames + 100)) + " 0"), num_frames);
  EXPECT_EQ(FramesModeEpochSize("-5 0"), 5);
}

// An end past the end of the video is clamped to the video's length.
TEST_F(VideoReaderDecoderFileListTest, FramesEndPastEndClamped_CpuOnlyTests) {
  int num_frames = cfr_videos_[0].NumFrames();
  EXPECT_EQ(FramesModeEpochSize("2 " + std::to_string(num_frames + 100)), num_frames - 2);
  // Beyond the range of int: must not wrap around to a negative (empty) end
  EXPECT_EQ(FramesModeEpochSize("2 3000000000"), num_frames - 2);
}

// A start at or past the end of the video is an error (same as timestamps mode).
TEST_F(VideoReaderDecoderFileListTest, FramesStartPastEndIsError_CpuOnlyTests) {
  int num_frames = cfr_videos_[0].NumFrames();
  // The last two are beyond the range of int: they must not wrap around to a negative frame
  // index or to frame 0.
  for (std::string start : {std::to_string(num_frames), std::to_string(num_frames + 100),
                            std::string("3000000000"), std::string("4294967300")}) {
    try {
      FramesModeEpochSize(start + " 0");
      FAIL() << "Expected an error for start=" << start << " (num_frames=" << num_frames << ")";
    } catch (const std::exception &e) {
      EXPECT_NE(std::string(e.what()).find("past the end of the video"), std::string::npos)
          << e.what();
    }
  }
}

// stride * sequence_length (the sequence extent and default step) must fit in int: before the
// fix, sequence_length=stride=65536 wrapped the default step to 0 and PrepareMetadataImpl looped
// forever.
TEST_F(VideoReaderDecoderCpuTest, SequenceExtentOverflowIsError_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 65536)
    .AddArg("stride", 65536)
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  try {
    pipe.Build({{"frames", "cpu"}});
    FAIL() << "Expected an error for stride * sequence_length overflowing int";
  } catch (const std::exception &e) {
    EXPECT_NE(std::string(e.what()).find("stride * sequence_length"), std::string::npos)
        << e.what();
  }
}

// A huge (but valid) step must not overflow the sample loop's frame counter: exactly one sample
// (starting at frame 0) fits per file.
TEST_F(VideoReaderDecoderCpuTest, HugeStepYieldsOneSamplePerFile_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("step", std::numeric_limits<int>::max())
    .AddArg("pad_mode", "constant")
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  pipe.Build({{"frames", "cpu"}});
  EXPECT_EQ(pipe.GetReaderMeta("r").epoch_size, 1);
}

TEST_F(VideoReaderDecoderCpuTest, DecoderCacheSizeMustBePositive_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("decoder_cache_size", 0)
    .AddArg("filenames", std::vector<std::string>{cfr_videos_paths_[0]})
    .AddOutput("frames", StorageDevice::CPU), "r");
  EXPECT_THROW(pipe.Build({{"frames", "cpu"}}), std::runtime_error);
}

// Eviction-boundary test for GetOrOpenDecoder's LRU cache, now sized via `decoder_cache_size`
// instead of the old hard-coded capacity. With decoder_cache_size=1, opening a second, different
// file must evict the first, so re-requesting the first file afterwards must build a fresh
// decoder (is_new_decoder=true) rather than reuse a cached one.
TEST_F(VideoReaderDecoderCpuTest, DecoderCacheSizeControlsEviction_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("decoder_cache_size", 1)
    .AddArg("filenames", cfr_videos_paths_)
    .AddOutput("frames", StorageDevice::CPU), "r");
  pipe.Build({{"frames", "cpu"}});

  auto *op = dynamic_cast<VideoReaderDecoder<CPUBackend> *>(pipe.GetOperator("r"));
  ASSERT_NE(op, nullptr);

  bool is_new_decoder = false;
  op->GetOrOpenDecoder(cfr_videos_paths_[0], is_new_decoder);
  EXPECT_TRUE(is_new_decoder);

  // Different file: with capacity 1, this evicts cfr_videos_paths_[0]'s entry.
  op->GetOrOpenDecoder(cfr_videos_paths_[1], is_new_decoder);
  EXPECT_TRUE(is_new_decoder);

  // Re-requesting the first file must rebuild it: it was evicted.
  op->GetOrOpenDecoder(cfr_videos_paths_[0], is_new_decoder);
  EXPECT_TRUE(is_new_decoder);
}

// Same as above, but with a cache large enough to hold both files: the second request for the
// first file must hit the cache (is_new_decoder=false).
TEST_F(VideoReaderDecoderCpuTest, DecoderCacheSizeAllowsReuseWhenLargeEnough_CpuOnlyTests) {
  Pipeline pipe(1, 1, CPU_ONLY_DEVICE_ID);
  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", 1)
    .AddArg("decoder_cache_size", 2)
    .AddArg("filenames", cfr_videos_paths_)
    .AddOutput("frames", StorageDevice::CPU), "r");
  pipe.Build({{"frames", "cpu"}});

  auto *op = dynamic_cast<VideoReaderDecoder<CPUBackend> *>(pipe.GetOperator("r"));
  ASSERT_NE(op, nullptr);

  bool is_new_decoder = false;
  op->GetOrOpenDecoder(cfr_videos_paths_[0], is_new_decoder);
  EXPECT_TRUE(is_new_decoder);

  op->GetOrOpenDecoder(cfr_videos_paths_[1], is_new_decoder);
  EXPECT_TRUE(is_new_decoder);

  // Both entries fit within capacity 2, so this must hit the cache.
  op->GetOrOpenDecoder(cfr_videos_paths_[0], is_new_decoder);
  EXPECT_FALSE(is_new_decoder);
}

class VideoReaderDecoderCompareTest : public VideoTestBase {};

TEST_F(VideoReaderDecoderCompareTest, CompareReaders) {
  const int batch_size = 4;
  const int sequence_length = 6;
  const int stride = 3;
  const int step = 10;
  const int shard_id = 3;
  const int num_shards = 4;
  const int seed = 1234;
  const int initial_fill = 50;

  Pipeline pipe(batch_size, 4, 0);

  pipe.AddOperator(OpSpec("experimental__readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg("stride", stride)
    .AddArg("step", step)
    .AddArg("shard_id", shard_id)
    .AddArg("num_shards", num_shards)
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
    .AddArg("shard_id", shard_id)
    .AddArg("num_shards", num_shards)
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
    .AddArg("shard_id", shard_id)
    .AddArg("num_shards", num_shards)
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

      auto height = cpu_sample_shape[1];
      auto width = cpu_sample_shape[2];

      for (int i = 0; i < sequence_length; ++i) {
        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          gpu_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        CompareFrameAvgError(i, frame_size, width, height, cpu_sample + i * frame_size,
                             frame_buffer.data(), eps);

        frame_buffer.clear();
        MemCopy(
          frame_buffer.data(),
          old_sample + i * frame_size,
          frame_size * sizeof(uint8_t));

        CompareFrameAvgError(i, frame_size, width, height, cpu_sample + i * frame_size,
                             frame_buffer.data(), eps);
      }
    }
  }
}
}  // namespace dali
