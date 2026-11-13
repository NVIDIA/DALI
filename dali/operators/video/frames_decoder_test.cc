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
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dali/core/call_at_exit.h"
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/video_test.h"
#include "dali/operators/video/video_utils.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/file.h"

#include "dali/pipeline/pipeline.h"

namespace dali {

// Reads the counters of the decoders that are not part of their interface (a friend of
// FramesDecoderBase)
struct FramesDecoderTestAccess {
  // The pictures that the decoder did not decode at all
  static int64_t NotDecodedPictures(const FramesDecoderBase &decoder) {
    return decoder.not_decoded_pictures_;
  }

  // The pictures that the decoder went through: those it decoded and those it left out (what
  // AddDecodeWork estimates)
  static int64_t ParsedPictures(const FramesDecoderBase &decoder) {
    return decoder.decoded_pictures_ + decoder.not_decoded_pictures_;
  }
};

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

// The video reader builds the index of the files with FramesIndexer, which only demuxes, instead of
// a decoder. It has to give the index that the decoder gives.
static void ExpectIndexerGivesTheIndexOfTheDecoder(FramesDecoderBase &decoder,
                                                   FramesIndexer &indexer) {
  ASSERT_TRUE(decoder.IsValid());
  decoder.BuildIndex();
  ASSERT_TRUE(indexer.IsValid());
  indexer.BuildIndex();

  const FrameIndex &expected = decoder.GetIndex();
  const FrameIndex &actual = indexer.GetIndex();
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(actual.filename, expected.filename);
  EXPECT_EQ(actual.timebase.num, expected.timebase.num);
  EXPECT_EQ(actual.timebase.den, expected.timebase.den);
  EXPECT_EQ(actual.decode_order_timestamps, expected.decode_order_timestamps);
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(actual.index[i].pts, expected.index[i].pts) << "frame " << i;
    EXPECT_EQ(actual.index[i].is_keyframe, expected.index[i].is_keyframe) << "frame " << i;
    EXPECT_EQ(actual.index[i].is_flush_frame, expected.index[i].is_flush_frame) << "frame " << i;
    EXPECT_EQ(actual.index[i].last_keyframe_id, expected.index[i].last_keyframe_id)
        << "frame " << i;
    EXPECT_EQ(actual.index[i].decode_idx, expected.index[i].decode_idx) << "frame " << i;
    EXPECT_EQ(actual.index[i].seek_decode_idx, expected.index[i].seek_decode_idx)
        << "frame " << i;
    EXPECT_EQ(actual.index[i].leading_pictures, expected.index[i].leading_pictures)
        << "frame " << i;
  }
  EXPECT_EQ(actual.end_pts, expected.end_pts);
  EXPECT_EQ(indexer.NumFrames(), decoder.NumFrames());
  EXPECT_EQ(indexer.Width(), decoder.Width());
  EXPECT_EQ(indexer.Height(), decoder.Height());
  EXPECT_EQ(indexer.IsVfr(), decoder.IsVfr());
}

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

// The timestamps of the packets where seeks to `timestamps` land, as AvSeekFrame seeks, asked to
// a demuxer of its own
std::vector<int64_t> SeekLandings(const std::string &path, const std::vector<int64_t> &timestamps) {
  AVUniquePtr<AVFormatContext> ctx;
  if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0) {
    ADD_FAILURE() << "Cannot open " << path;
    return {};
  }
  int stream = -1;
  if (avformat_find_stream_info(ctx, nullptr) >= 0)
    stream = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream < 0) {
    ADD_FAILURE() << "No video stream in " << path;
    return {};
  }
  AVUniquePtr<AVPacket> packet(av_packet_alloc());
  std::vector<int64_t> landings;
  for (int64_t ts : timestamps) {
    int64_t landing = AV_NOPTS_VALUE;
    if (av_seek_frame(ctx, stream, ts, AVSEEK_FLAG_BACKWARD) >= 0) {
      while (av_read_frame(ctx, packet) == 0) {
        bool found = packet->stream_index == stream;
        if (found)
          landing = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
        av_packet_unref(packet);
        if (found)
          break;
      }
    }
    landings.push_back(landing);
  }
  return landings;
}

// Checks the decoding order in the index of a file without hidden packets, where a seek to a
// keyframe lands on the keyframe itself (BuildIndex records this without asking the demuxer, so
// it is asked here)
void CheckDecodeOrder(const FrameIndex &index, const std::string &path) {
  int n = static_cast<int>(index.size());
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
      // a target that is not a keyframe (frame 0 before the first one): the start of the stream
      int landing = index[i].is_keyframe ? index[i].decode_idx : 0;
      EXPECT_EQ(index[i].seek_decode_idx, landing) << "seek to frame " << i;
    } else {
      EXPECT_EQ(index[i].seek_decode_idx, -1) << "frame " << i << " is not a seek target";
    }
  }
  std::vector<int64_t> timestamps;
  for (int target : targets)
    timestamps.push_back(index[target].pts);
  auto landings = SeekLandings(path, timestamps);
  ASSERT_EQ(landings.size(), timestamps.size());
  for (size_t i = 0; i < landings.size(); i++)
    EXPECT_EQ(landings[i], timestamps[i]) << path << ": a seek to " << timestamps[i];
}

std::string SintelVideoFile(int clip) {
  return make_string(testing::dali_extra_path(), "/db/video/sintel/video_files/",
                     "sintel_trailer-720p_", clip, ".mp4");
}

// One Sintel clip of each kind, all with B-frames: no lead-in, only the IDR hidden (a single GOP),
// a seek that lands on the hidden IDR, seeks that land on the shown IDRs, and a long lead-in (188
// hidden pictures)
std::vector<std::string> SintelClipKinds() {
  return {SintelVideoFile(0), SintelClip(0, 3), SintelClip(0, 9), SintelClip(2, 5),
          SintelClip(2, 8)};
}

constexpr int kMaxDecodeWorkFrames = 32;  // in a decode work scenario

std::vector<int> FrameRange(int start, int end) {
  std::vector<int> frames;
  for (int i = start; i < end; i++)
    frames.push_back(i);
  return frames;
}

// How many frames the stream may show after a later one in decoding order (its reordering depth),
// by codec. The test clips come from x264 and x265 with their default settings (B-frames with a
// B-pyramid: 2 pictures reordered, the SPS of the HEVC clip says the same); VP9 does not reorder.
// This is not taken from libavformat (AVCodecParameters::video_delay): it is an estimate that
// differs between FFmpeg versions, and a bound of this test must not depend on the FFmpeg build.
int StreamReorderDepth(const std::string &path) {
  AVUniquePtr<AVFormatContext> ctx;
  if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0) {
    ADD_FAILURE() << "Cannot open " << path;
    return 0;
  }
  int stream = -1;
  if (avformat_find_stream_info(ctx, nullptr) >= 0)
    stream = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream < 0) {
    ADD_FAILURE() << "No video stream in " << path;
    return 0;
  }
  switch (ctx->streams[stream]->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
    case AV_CODEC_ID_HEVC:
      return 2;
    default:
      return 0;
  }
}

// A GPU decoder is ahead of the estimate of AddDecodeWork (the pictures up to the one of the last
// frame read): it has decoded the pictures that the frames shown next need, as many frames as the
// parser holds back. Frames are displayed in order, so it holds back as many decoded pictures as
// the stream may reorder (StreamReorderDepth), and on top of it kDisplayDelay. A packet is decoded
// when the next one arrives, and now and then (e.g. some B-frames of the Sintel clips) only with
// the one after: the packet lag, one frame. The lead is there from the first frame after a start,
// and shrinks only near the end of the stream, where the decoder cannot decode past the last
// picture. The "lookahead" below is that number of frames; a CPU decoder has none.
// The constants of this model were measured, not derived from the NVDEC parser: the lead was the
// same on a TITAN V and an RTX 5080 (drivers 580 and 620), and the CI runs with an L40 and an H100
// failed with the same counts as the old slack did on those two. Some clips meet the bound
// exactly, with no headroom.
constexpr int kPacketLag = 1;

int GpuLookahead(const std::string &path) {
  return kPacketLag + StreamReorderDepth(path) + FramesDecoderGpu::kDisplayDelay;
}

/**
 * @brief The most pictures that a decoder may have decoded beyond the estimate at `pos`: the
 * pictures of the next `lookahead` frames (see kPacketLag), which is none at the end of the stream.
 */
int64_t LeadAllowance(const FrameIndex &index, FramesDecoderBase::DecodePosition pos,
                      int lookahead) {
  if (pos.next_frame < 0)
    return 0;
  int end = std::min(pos.next_frame + lookahead, static_cast<int>(index.size()));
  FramesDecoderBase::DecodeWork work;
  for (int frame = pos.next_frame; frame < end; frame++)
    FramesDecoderBase::AddDecodeWork(index, pos, frame, work);
  return work.pictures;
}

// How many pictures a decoder has decoded beyond the estimate, after a sample: between `min` and
// `max`. It is exact (`min == max`) unless a seek in the middle of a sample hides it.
struct DecodeLead {
  int64_t min = 0, max = 0;
};

/**
 * @brief Decodes `frame_ids` as the reader does (padded by clamping) and checks that the decoder
 * goes through the pictures that AddDecodeWork estimates from `pos` (those it
 * decodes and those it leaves out), reopens the file as estimated, and ends where `pos` does.
 *
 * The decoder may be ahead of the estimate by the pictures of the next `lookahead` frames (see
 * kPacketLag). `lead` carries how far ahead it is from the previous samples: the next frames
 * do not cost those pictures. It is not clamped, as the lead is lost only near the end of the
 * stream, where the decoder cannot decode past the last picture (LeadAllowance shrinks to none
 * there). A seek or a reset drops the lead. Neither may the decoder be behind the estimate by more
 * than the lead that it had.
 *
 * A sample takes as many leads as the segments between its seeks (the second of two equal frames
 * seeks back), and only the sum of their changes is observed: with a seek in the middle of a
 * sample, the lead at its end is known only to be within [0, allowance].
 */
void CheckDecodeWork(FramesDecoderBase &decoder, FramesDecoderBase::DecodePosition &pos,
                     DecodeLead &lead, const std::vector<int> &frame_ids, uint8_t *buffer,
                     int lookahead, const std::string &what) {
  ASSERT_LE(frame_ids.size(), static_cast<size_t>(kMaxDecodeWorkFrames));
  auto &index = decoder.GetIndex();
  int n = static_cast<int>(index.size());
  // the frames that DecodeFrames seeks to, in order (keep in sync with DecodeFrames)
  std::vector<int> frames;
  for (int frame : frame_ids)
    frames.push_back(FramesDecoderBase::HandleBoundary(boundary::BoundaryType::CLAMP, frame, 0, n));
  std::sort(frames.begin(), frames.end());
  // A seek or a reset drops the pictures decoded ahead
  auto restarts = [&](const FramesDecoderBase::DecodePosition &at, int frame) {
    return at.next_frame < 0 || at.next_frame >= n ||
           (frame != at.next_frame &&
            FramesDecoderBase::SeekTarget(index, at.next_frame, frame) >= 0);
  };
  if (restarts(pos, frames[0]))
    lead = {0, 0};
  // The leads that the segments of the sample may end with, in total: each is bounded by its own
  // allowance
  int64_t allowed = 0;
  bool seeks_inside = false;
  FramesDecoderBase::DecodeWork work;
  for (size_t i = 0; i < frames.size(); i++) {
    if (i > 0 && restarts(pos, frames[i])) {
      allowed += LeadAllowance(index, pos, lookahead);  // a segment ends here
      seeks_inside = true;
    }
    FramesDecoderBase::AddDecodeWork(index, pos, frames[i], work);
  }
  int64_t end_allowed = LeadAllowance(index, pos, lookahead);
  allowed += end_allowed;

  int64_t before = FramesDecoderTestAccess::ParsedPictures(decoder);
  int64_t reopens_before = decoder.Reopens();
  decoder.DecodeFrames(buffer, make_cspan(frame_ids), boundary::BoundaryType::CLAMP);
  int64_t ahead = FramesDecoderTestAccess::ParsedPictures(decoder) - before - work.pictures;
  EXPECT_GE(ahead, -lead.max) << what << ": the decoder is behind the estimate by more than it "
                              << "was ahead";
  EXPECT_LE(ahead, allowed - lead.min)
      << what << ": the decoder is further ahead of the estimate than " << lookahead << " frames";
  if (seeks_inside) {
    lead = {0, end_allowed};
  } else {
    lead.min = std::max<int64_t>(lead.min + ahead, 0);
    lead.max = std::min<int64_t>(lead.max + ahead, end_allowed);
  }
  EXPECT_EQ(decoder.Reopens() - reopens_before, work.reopens) << what;
  EXPECT_EQ(decoder.NextFrameIdx(), pos.next_frame) << what;
}

/**
 * @brief Runs, on one decoder, samples that take every path of SeekFrame: decoding through,
 * seeking forward and backward, to the same and to another keyframe, and resetting.
 *
 * A file without a keyframe early enough (e.g. a single GOP) skips the sample across a keyframe;
 * `across_keyframe` counts the files that ran it.
 */
void CheckDecodeWorkScenarios(FramesDecoderBase &decoder, uint8_t *buffer, int lookahead,
                              const std::string &path, int &across_keyframe) {
  auto &index = decoder.GetIndex();
  int n = static_cast<int>(index.size());
  ASSERT_GE(n, 40);
  FramesDecoderBase::DecodePosition pos;  // a new decoder
  DecodeLead lead;
  int a = std::min(kMaxDecodeWorkFrames, n / 5);
  int next = a;
  auto check = [&](const std::string &what, const std::vector<int> &frames) {
    ASSERT_NO_FATAL_FAILURE(CheckDecodeWork(decoder, pos, lead, frames, buffer, lookahead,
                                            make_string(path, ": ", what)));
    next = pos.next_frame;
  };
  check("fresh", FrameRange(0, a));
  check("continue", FrameRange(next, next + 8));
  constexpr int kShortLeap = 5, kShortLeapEnd = 9, kFarJump = 20, kTail = 4;
  // Early in the file, so that there is a keyframe ahead and room for the leaps after it
  int keyframe = next + 1;
  while (keyframe < n && !index[keyframe].is_keyframe)
    keyframe++;
  int after_keyframe = keyframe + 4;  // where the sample across the keyframe ends
  if (after_keyframe + kShortLeapEnd < n - kTail) {
    check("across a keyframe", FrameRange(keyframe + 1, after_keyframe));
    across_keyframe++;
  }
  static_assert(FramesDecoderBase::kMinimumSeekLeap >= kShortLeap);
  check("short leap", FrameRange(next + kShortLeap, next + kShortLeapEnd));
  // not to the last frame: the next sample seeks backward
  int far = std::min(next + kFarJump, n - kTail);
  check("far jump", FrameRange(far, far + 3));
  check("backward", FrameRange(5, 9));
  check("duplicate", {12, 12, 13});
  check("last frame, twice", {n - 1, n - 1});
  check("first frame after the end", {0, 1});
  std::vector<int> uniform;
  for (int i = 0; i < 8; i++)
    uniform.push_back(std::lround(i * (n - 1) / 7.0));
  check("uniform", uniform);
  check("edge-padded tail", FrameRange(n - kTail, n + 3));
  check("from the start", FrameRange(0, 4));
}

/**
 * @brief Reads the frames in a row and checks that, from PositionAt, AddDecodeWork estimates the
 * pictures gone through (FramesDecoderTestAccess::ParsedPictures) for each frame: in total, the decoder is never behind the
 * estimate and at most the pictures of the next `lookahead` frames ahead of it (see kPacketLag).
 * Returns the number of frames that cost nothing (decoded earlier, for a frame shown before them).
 */
int CheckPositionAt(FramesDecoderBase &decoder, uint8_t *frame, int lookahead,
                    const std::string &path) {
  auto &index = decoder.GetIndex();
  int free_frames = 0;
  int64_t start = FramesDecoderTestAccess::ParsedPictures(decoder), estimate = 0;
  for (int k = 0; k < static_cast<int>(index.size()); k++) {
    auto pos = FramesDecoderBase::PositionAt(index, k);
    FramesDecoderBase::DecodeWork work;
    FramesDecoderBase::AddDecodeWork(index, pos, k, work);
    estimate += work.pictures;
    EXPECT_EQ(decoder.NextFrameIdx(), k) << path;
    decoder.ReadNextFrame(frame);
    int64_t ahead = FramesDecoderTestAccess::ParsedPictures(decoder) - start - estimate;
    EXPECT_GE(ahead, 0) << path << ": frame " << k;
    EXPECT_LE(ahead, LeadAllowance(index, pos, lookahead)) << path << ": frame " << k;
    free_frames += work.pictures == 0;
  }
  return free_frames;
}

// An index built by hand: the decoding position of each frame (-1: unknown), its keyframes, and
// where seeks to some of them land
FrameIndex MakeIndex(const std::vector<int> &decode_idx, const std::set<int> &keyframes,
                     const std::map<int, int> &seek_decode_idx = {}) {
  FrameIndex index = {};
  int last_keyframe = 0;
  for (int i = 0; i < static_cast<int>(decode_idx.size()); i++) {
    IndexEntry entry = {};
    entry.pts = i;
    entry.is_keyframe = keyframes.count(i) > 0;
    if (entry.is_keyframe)
      last_keyframe = i;
    entry.last_keyframe_id = last_keyframe;
    entry.is_flush_frame = false;
    entry.decode_idx = decode_idx[i];
    index.index.push_back(entry);
  }
  for (auto [frame, landing] : seek_decode_idx)
    index[frame].seek_decode_idx = landing;
  return index;
}

struct DecodeStep {
  int frame;
  int64_t pictures;  // decoded to return it
  int reopens;
  int next_frame;  // after it
};

void CheckDecodeSteps(const FrameIndex &index, const std::vector<DecodeStep> &steps) {
  FramesDecoderBase::DecodePosition pos;
  for (auto &step : steps) {
    FramesDecoderBase::DecodeWork work;
    FramesDecoderBase::AddDecodeWork(index, pos, step.frame, work);
    EXPECT_EQ(work.pictures, step.pictures) << "frame " << step.frame;
    EXPECT_EQ(work.reopens, step.reopens) << "frame " << step.frame;
    EXPECT_EQ(pos.next_frame, step.next_frame) << "frame " << step.frame;
  }
}

// PositionAt gives the position of a decoder that read the frames before it in a row
void CheckPositionAtInARow(const FrameIndex &index) {
  int n = static_cast<int>(index.size());
  FramesDecoderBase::DecodePosition pos;
  for (int k = 0; k < n; k++) {
    auto at = FramesDecoderBase::PositionAt(index, k);
    EXPECT_EQ(at.next_frame, pos.next_frame) << "frame " << k;
    EXPECT_EQ(at.decoded, pos.decoded) << "frame " << k;
    FramesDecoderBase::DecodeWork work;
    FramesDecoderBase::AddDecodeWork(index, pos, k, work);
  }
  EXPECT_EQ(FramesDecoderBase::PositionAt(index, n).next_frame, -1);
  EXPECT_EQ(FramesDecoderBase::PositionAt(index, -1).next_frame, -1);
}

}  // namespace

static_assert(FramesDecoderBase::kMinimumSeekLeap == 10, "the steps below assume it");

// Exact estimates on hand-built indices: B-frames, a hidden lead-in, a seek that lands before its
// keyframe and an open GOP, which the decoder tests only cover with real files and some lookahead
TEST(FramesDecoderDecodeWorkTest, ReorderedWithLeadIn) {
  // 8 GOPs of 4 frames, shown as I B B P and decoded as I P B B, after 2 hidden packets. A seek to
  // frame 0 lands on the first hidden packet, and a seek to frame 16 on the keyframe before it.
  std::vector<int> decode_idx;
  std::set<int> keyframes;
  for (int g = 0; g < 8; g++) {
    int d = 2 + 4 * g;
    decode_idx.insert(decode_idx.end(), {d, d + 2, d + 3, d + 1});
    keyframes.insert(4 * g);
  }
  auto index = MakeIndex(decode_idx, keyframes, {{0, 0}, {16, 14}});
  EXPECT_EQ(index.SeekDecodeIdx(0), 0);
  EXPECT_EQ(index.SeekDecodeIdx(16), 14);
  EXPECT_EQ(index.SeekDecodeIdx(20), 22);  // unknown: the keyframe itself
  CheckDecodeSteps(index, {
    {0, 3, 0, 1},     // the hidden packets and the I-frame
    {1, 2, 0, 2},     // the P-frame and the first B-frame
    {2, 1, 0, 3},
    {3, 0, 0, 4},     // the P-frame, already decoded
    {6, 4, 0, 7},     // decoding through, from frame 4
    {20, 1, 0, 21},   // a far jump, to a keyframe
    {17, 7, 0, 18},   // backward: the seek to frame 16 lands on frame 12
    {31, 4, 0, -1},   // the last frame
    {31, 4, 1, -1},   // again: reopens the file, then seeks to frame 28
    {2, 6, 1, 3},     // after the end: reopens the file and decodes from the start
    {-1, 0, 0, 3},    // a constant frame: not decoded
    {32, 0, 0, 3},    // past the end: not decoded
    {13, 11, 0, 14},  // a leap of kMinimumSeekLeap frames: decoding through 2 keyframes
    {15, 1, 0, 16},
  });
  CheckPositionAtInARow(index);
  EXPECT_EQ(FramesDecoderBase::PositionAt(index, 6).decoded, 8);
}

TEST(FramesDecoderDecodeWorkTest, OpenGop) {
  // Shown as I P B B I P, decoded as I P I B B P: the B-frames are decoded after the second
  // keyframe, and a decoder that read them is past it
  auto index = MakeIndex({0, 1, 3, 4, 2, 5}, {0, 4});
  CheckDecodeSteps(index, {{0, 1, 0, 1}, {1, 1, 0, 2}, {2, 2, 0, 3}, {3, 1, 0, 4}, {4, 0, 0, 5},
                           {5, 1, 0, -1}});
  CheckPositionAtInARow(index);
  EXPECT_EQ(FramesDecoderBase::PositionAt(index, 5).decoded, 4);
}

// The decision that SeekFrame and AddDecodeWork share
TEST(FramesDecoderDecodeWorkTest, SeekTarget) {
  // GOPs of 15 frames: keyframes 0, 15, 30
  auto index = MakeIndex(std::vector<int>(45, -1), {0, 15, 30});
  auto SeekTarget = [&](int next, int frame) {
    return FramesDecoderBase::SeekTarget(index, next, frame);
  };
  EXPECT_EQ(SeekTarget(3, 3), -1);    // already there
  EXPECT_EQ(SeekTarget(3, 13), -1);   // a leap of kMinimumSeekLeap
  EXPECT_EQ(SeekTarget(3, 14), -1);   // further, but the same keyframe
  EXPECT_EQ(SeekTarget(10, 20), -1);  // a leap of kMinimumSeekLeap, across a keyframe
  EXPECT_EQ(SeekTarget(3, 16), 15);   // further, to another keyframe
  EXPECT_EQ(SeekTarget(3, 40), 30);
  EXPECT_EQ(SeekTarget(14, 13), 0);   // backward, within the same keyframe
  EXPECT_EQ(SeekTarget(20, 5), 0);    // backward, to another keyframe
}

// With the timestamps in decoding order (AVI), the frames right after a keyframe may be its leading
// pictures, which need the pictures before it: SeekFrame seeks to the keyframe before
TEST(FramesDecoderDecodeWorkTest, SeekTargetWithDecodeOrderTimestamps) {
  static_assert(FramesDecoderBase::kMaxLeadingPictures == 16, "the frames below assume it");
  // GOPs of 40 frames: keyframes 0, 40, 80
  auto index = MakeIndex(std::vector<int>(120, -1), {0, 40, 80});
  index.decode_order_timestamps = true;
  auto SeekTarget = [&](int next, int frame) {
    return FramesDecoderBase::SeekTarget(index, next, frame);
  };
  EXPECT_EQ(SeekTarget(3, 5), -1);     // a short leap
  EXPECT_EQ(SeekTarget(30, 5), 0);     // backward, in the first GOP: no keyframe before
  EXPECT_EQ(SeekTarget(3, 41), -1);    // right after keyframe 40: decoding through from GOP 0
  EXPECT_EQ(SeekTarget(70, 41), 0);    // backward, right after keyframe 40: the keyframe before
  EXPECT_EQ(SeekTarget(70, 55), 0);    // the last frame that may lead keyframe 40
  EXPECT_EQ(SeekTarget(70, 56), 40);   // past the leading pictures
  EXPECT_EQ(SeekTarget(3, 56), 40);
  EXPECT_EQ(SeekTarget(20, 85), 40);   // right after keyframe 80: seeks to keyframe 40
  EXPECT_EQ(SeekTarget(45, 85), -1);   // ... where the decoder already is: decoding through
  EXPECT_EQ(SeekTarget(30, 90), 40);   // a longer leap from GOP 0
  CheckPositionAtInARow(index);

  // Known leading pictures (MPEG-4 Part 2): 2 for keyframe 40, none for keyframe 80
  index[40].leading_pictures = 2;
  index[80].leading_pictures = 0;
  EXPECT_EQ(SeekTarget(70, 40), 0);    // a leading picture: the keyframe before
  EXPECT_EQ(SeekTarget(70, 41), 0);
  EXPECT_EQ(SeekTarget(70, 42), 40);   // the keyframe itself, shown after its leading pictures
  EXPECT_EQ(SeekTarget(70, 55), 40);
  EXPECT_EQ(SeekTarget(20, 80), 80);   // no leading pictures
  EXPECT_EQ(SeekTarget(20, 85), 80);
  CheckPositionAtInARow(index);

  index.decode_order_timestamps = false;  // the same index, with presentation timestamps
  EXPECT_EQ(SeekTarget(70, 41), 40);
  EXPECT_EQ(SeekTarget(3, 41), 40);
  EXPECT_EQ(SeekTarget(20, 85), 80);
}

TEST(FramesDecoderDecodeWorkTest, NoDecodeOrder) {
  // One GOP of 30 frames, without decoding positions: taken in display order
  auto index = MakeIndex(std::vector<int>(30, -1), {0});
  EXPECT_EQ(index.DecodeIdx(7), 7);
  EXPECT_EQ(index.SeekDecodeIdx(0), 0);
  CheckDecodeSteps(index, {
    {0, 1, 0, 1},
    {20, 20, 0, 21},  // past kMinimumSeekLeap, but the same keyframe: decoding through
    {5, 6, 0, 6},     // backward: seeks to frame 0
  });
  CheckPositionAtInARow(index);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, DecodeWork) {
  int across_keyframe = 0;
  auto paths = cfr_videos_paths_;
  paths.insert(paths.end(), vfr_videos_paths_.begin(), vfr_videos_paths_.end());
  for (auto &path : paths) {
    FramesDecoderCpu decoder(path);
    decoder.BuildIndex();
    std::vector<uint8_t> buffer(kMaxDecodeWorkFrames * decoder.FrameSize());
    ASSERT_NO_FATAL_FAILURE(
        CheckDecodeWorkScenarios(decoder, buffer.data(), 0, path, across_keyframe));
    decoder.Reset();
    CheckPositionAt(decoder, buffer.data(), 0, path);
  }
  EXPECT_GT(across_keyframe, 0) << "no file ran the sample across a keyframe";
}

TEST_F(FramesDecoderGpuTest, DecodeWork) {
  auto with_b_frames = SintelClipKinds();
  auto paths = with_b_frames;
  // VP9, H.264 without a lead-in and HEVC (the CPU test covers the other VP9 files)
  paths.push_back(vfr_videos_paths_[0]);
  paths.push_back(make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4"));
  if (FramesDecoderGpu::SupportsHevc())
    paths.push_back(vfr_hevc_videos_paths_[0]);
  int across_keyframe = 0;
  for (size_t i = 0; i < paths.size(); i++) {
    auto &path = paths[i];
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    DeviceBuffer<uint8_t> buffer;
    buffer.resize(kMaxDecodeWorkFrames * decoder.FrameSize());
    int lookahead = GpuLookahead(path);
    ASSERT_NO_FATAL_FAILURE(
        CheckDecodeWorkScenarios(decoder, buffer.data(), lookahead, path, across_keyframe));
    decoder.Reset();
    int free_frames = CheckPositionAt(decoder, buffer.data(), lookahead, path);
    if (i < with_b_frames.size()) {
      EXPECT_GT(free_frames, 0) << path << ": no B-frames";
    }
  }
  EXPECT_GT(across_keyframe, 0) << "no file ran the sample across a keyframe";
}

TEST_F(FramesDecoderTest_CpuOnlyTests, IndexDecodeOrder) {
  for (auto &path : {cfr_videos_paths_[0], vfr_videos_paths_[0]}) {
    FramesDecoderCpu decoder(path);
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    CheckDecodeOrder(index, path);
    for (size_t i = 0; i < index.size(); i++)  // VP9 has no reordering
      EXPECT_EQ(index[i].decode_idx, static_cast<int>(i)) << path;
  }
}

// The end of the stream recorded by BuildIndex is used, not the estimate from the last gap
// (the clip has a variable frame rate: the recorded end is 30464, the estimate 30976)
TEST_F(FramesDecoderTest_CpuOnlyTests, EndPtsIsTheRecordedEndOfTheStream) {
  FramesDecoderCpu decoder(vfr_videos_paths_[0]);
  decoder.BuildIndex();
  auto &index = decoder.GetIndex();
  ASSERT_EQ(index.size(), 50u);
  EXPECT_EQ(index.end_pts, 30464);
  EXPECT_EQ(index.EndPts(), 30464);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(30464, true), 50);  // the end: one past the last frame
  EXPECT_EQ(index.GetFrameIdxByTimestamp(30463, true), 49);  // inside the last frame
}

// H.264, with reordering (the index of the VP9 files does not depend on the decoder: see the CPU
// test)
TEST_F(FramesDecoderGpuTest, IndexDecodeOrder) {
  auto path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4");
  FramesDecoderGpu decoder(path);
  decoder.BuildIndex();
  auto &index = decoder.GetIndex();
  CheckDecodeOrder(index, path);
  int reordered = 0;
  for (size_t i = 0; i < index.size(); i++)
    reordered += index[i].decode_idx != static_cast<int>(i);
  EXPECT_GT(reordered, 0) << "the decoding order is the display order";
}

// The Sintel clips start with packets that an edit list hides (negative timestamps): they are not
// in the index, but they are decoded, before the first frame or after a seek that lands on them
TEST_F(FramesDecoderGpuTest, IndexDecodeOrderLeadIn) {
  {
    // 310 packets, of which 188 hidden, and only the IDR at frame 62 is shown
    auto path = SintelClip(2, 8);
    FramesDecoderGpu decoder(path);
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
    EXPECT_EQ(SeekLandings(path, {index[62].pts}), std::vector<int64_t>{index[62].pts});
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
    // Here the seeks to the shown IDRs land on them (as a demuxer of the test's own confirms: the
    // index would say so too if BuildIndex failed to ask)
    auto path = SintelClip(2, 5);
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    auto &index = decoder.GetIndex();
    auto landings = SeekLandings(path, {index[21].pts, index[58].pts});
    ASSERT_EQ(landings.size(), 2u);
    EXPECT_EQ(landings[0], index[21].pts);
    EXPECT_EQ(landings[1], index[58].pts);
    for (int k : {21, 58}) {
      ASSERT_TRUE(index[k].is_keyframe) << k;
      EXPECT_EQ(index.SeekDecodeIdx(k), index.DecodeIdx(k)) << k;
    }
    EXPECT_EQ(index.DecodeIdx(21), 37);
    EXPECT_EQ(index.DecodeIdx(58), 74);
  }
}

namespace {

// A decoder that decodes every picture, and maps and converts every picture that it buffers (see
// FramesDecoderGpu::BufferedFrameData): it leaves out none of the pictures that are not read. It
// takes the frame buffers for them as the default decoder does (AttachFrameBuffer).
class GpuDecoderMappingAll : public FramesDecoderGpu {
 public:
  template <typename... Args>
  explicit GpuDecoderMappingAll(Args &&...args) : FramesDecoderGpu(std::forward<Args>(args)...) {
    SetBufferedFrameDataForTesting(BufferedFrameData::kAll);
  }
};

// A decoder that maps none of the pictures that it buffers, and decodes an H.264 non-reference
// picture only if it is the next frame, read with its data, when it is decoded
class GpuDecoderMappingNone : public FramesDecoderGpu {
 public:
  template <typename... Args>
  explicit GpuDecoderMappingNone(Args &&...args) : FramesDecoderGpu(std::forward<Args>(args)...) {
    SetBufferedFrameDataForTesting(BufferedFrameData::kNone);
  }
};

// The pictures that a decoder maps past the last frame it reads: those shown after it that the
// parser displays in the same cuvidParseVideoData call as that frame (or, at the end of the stream,
// all those it holds), which a next sample of a decode lane may read
constexpr int kMaxMappedAhead = 4;

// The stream of the decoders of these tests: the default one
constexpr cudaStream_t kStream = 0;

// Hashes of the `num_frames` frames in `frames` (on the device)
std::vector<size_t> HashFrames(const uint8_t *frames, int num_frames, size_t frame_size,
                               cudaStream_t stream) {
  std::vector<char> host(frame_size);
  std::vector<size_t> hashes;
  for (int i = 0; i < num_frames; i++) {
    MemCopy(host.data(), frames + i * frame_size, frame_size, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));
    hashes.push_back(std::hash<std::string_view>()(std::string_view(host.data(), frame_size)));
  }
  return hashes;
}

std::vector<int> Strided(int start, int end, int stride) {
  std::vector<int> frames;
  for (int i = start; i < end; i += stride)
    frames.push_back(i);
  return frames;
}

// Decodes `frame_ids` with DecodeFrames (padding with a constant frame) and hashes them
std::vector<size_t> DecodeAndHash(
    FramesDecoderGpu &decoder, const std::vector<int> &frame_ids,
    boundary::BoundaryType boundary_type = boundary::BoundaryType::ISOLATED) {
  size_t frame_size = decoder.FrameSize();
  DeviceBuffer<uint8_t> frames, constant;
  frames.resize(frame_ids.size() * frame_size);
  constant.resize(frame_size);
  CUDA_CALL(cudaMemsetAsync(constant.data(), 77, frame_size, kStream));
  decoder.DecodeFrames(frames.data(), make_cspan(frame_ids), boundary_type, constant.data());
  return HashFrames(frames.data(), frame_ids.size(), frame_size, kStream);
}

// Reads `frame_ids` in this order with ReadNextFrame, seeking to a frame that is not the next one,
// and hashes them
std::vector<size_t> ReadAndHash(FramesDecoderGpu &decoder, const std::vector<int> &frame_ids) {
  DeviceBuffer<uint8_t> frames;
  frames.resize(frame_ids.size() * decoder.FrameSize());
  uint8_t *out = frames.data();
  for (int frame : frame_ids) {
    if (decoder.NextFrameIdx() != frame)
      decoder.SeekFrame(frame);
    EXPECT_TRUE(decoder.ReadNextFrame(out)) << "frame " << frame;
    out += decoder.FrameSize();
  }
  return HashFrames(frames.data(), frame_ids.size(), decoder.FrameSize(), kStream);
}

using HashedFramesScenario = std::function<std::vector<size_t>(FramesDecoderGpu &)>;

struct ScenarioResult {
  std::vector<size_t> frames;
  int64_t decoded = 0, not_decoded = 0, mapped = 0;
};

// Runs `scenario` with a new decoder
template <typename Decoder>
ScenarioResult RunScenario(const std::string &path, const FrameIndex &index,
                           const HashedFramesScenario &scenario) {
  Decoder decoder(path);
  decoder.SetIndex(index);
  ScenarioResult result;
  result.frames = scenario(decoder);
  result.decoded = decoder.DecodedPictures();
  result.not_decoded = FramesDecoderTestAccess::NotDecodedPictures(decoder);
  result.mapped = decoder.MappedPictures();
  return result;
}

// A scenario run with a decoder (`result`) and with a GpuDecoderMappingAll (`expected`)
struct ComparedScenario {
  ScenarioResult result, expected;
};

/**
 * @brief Runs `scenario` with a decoder and with one that decodes every picture and maps every
 * picture that it buffers (GpuDecoderMappingAll), and checks that they return the same frames,
 * after going through the same pictures, of which the decoder decodes and maps no more.
 */
ComparedScenario CheckSameFramesAsMappingAll(const std::string &path, const FrameIndex &index,
                                             const std::string &what,
                                             const HashedFramesScenario &scenario) {
  auto result = RunScenario<FramesDecoderGpu>(path, index, scenario);
  auto expected = RunScenario<GpuDecoderMappingAll>(path, index, scenario);
  EXPECT_EQ(result.frames, expected.frames) << path << ": " << what;
  EXPECT_EQ(result.decoded + result.not_decoded, expected.decoded + expected.not_decoded)
      << path << ": " << what;
  EXPECT_EQ(expected.not_decoded, 0) << path << ": " << what << ": the reference left some out";
  EXPECT_LE(result.decoded, expected.decoded) << path << ": " << what;
  EXPECT_LE(result.mapped, expected.mapped) << path << ": " << what;
  return {result, expected};
}

/**
 * @brief Hashes of `frames`, read in a row by a new decoder with their data, from the first frame
 * up to the last of `frames`.
 *
 * The decoder decodes every picture and maps every picture that it buffers
 * (GpuDecoderMappingAll), and it reads every frame in a row: these frames do not depend on which
 * pictures a decoder leaves out when it skips frames, or reads them after a seek.
 */
std::map<int, size_t> SequentialFrameHashes(const std::string &path, const FrameIndex &index,
                                            const std::set<int> &frames) {
  GpuDecoderMappingAll decoder(path);
  decoder.SetIndex(index);
  size_t frame_size = decoder.FrameSize();
  DeviceBuffer<uint8_t> frame;
  frame.resize(frame_size);
  std::map<int, size_t> hashes;
  for (int i = 0; i <= *frames.rbegin(); i++) {
    EXPECT_EQ(decoder.NextFrameIdx(), i) << path;
    EXPECT_TRUE(decoder.ReadNextFrame(frame.data())) << path << ": frame " << i;
    if (frames.count(i))
      hashes[i] = HashFrames(frame.data(), 1, frame_size, kStream)[0];
  }
  return hashes;
}

/**
 * @brief Checks that a decoder that does not map the pictures it does not read returns the same
 * frames as GpuDecoderMappingAll, when it decodes samples (with skipped frames, duplicates and
 * padding), when it seeks, also with AvSeekFrame to a frame that is not a keyframe, and when one
 * decoder decodes a sample after another (a decode lane), also one that starts before the end of
 * the previous one.
 *
 * GpuDecoderMappingAll differs from the decoder only in which pictures it leaves out, so the frames
 * of the samples within the clip (not padded) are also compared with those of a sequential read
 * (SequentialFrameHashes); except after AvSeekFrame to a frame that is not a keyframe, where the
 * demuxer lands decides which frames come.
 *
 * @param skips_non_reference The clip has H.264 non-reference pictures among the frames that the
 *        "stride 3" and "seeks" samples skip (not in a lead-in): the decoder must not decode them.
 */
void CheckSameFramesAsMappingAll(const std::string &path, bool skips_non_reference = false) {
  FramesDecoderGpu indexed(path);
  indexed.BuildIndex();
  auto &index = indexed.GetIndex();
  int n = index.size();
  ASSERT_GE(n, 40) << path;
  bool is_h264 = FramesIndexer(path).CodecId() == AV_CODEC_ID_H264;
  static_assert(FramesDecoderBase::kMinimumSeekLeap >= 3);
  constexpr int leap = FramesDecoderBase::kMinimumSeekLeap;
  const auto stride_3 = Strided(0, 40, 3);
  const std::vector<int> duplicate = {5, 5, 7, 30, 31, std::min(90, n - 1)};
  const std::vector<int> seeks = {0, leap, 2 * leap + 1, 2 * leap + 3, n / 2, n / 2 + 3, 1};
  const std::vector<std::vector<int>> lane = {Strided(0, 9, 2), FrameRange(9, 17),
                                              FrameRange(16, 20)};
  const std::vector<int> lane_frames = [&]() {
    std::vector<int> frames;
    for (auto &sample : lane)
      frames.insert(frames.end(), sample.begin(), sample.end());
    return frames;
  }();

  std::set<int> in_clip;
  for (auto *frames : {&stride_3, &duplicate, &seeks, &lane_frames})
    in_clip.insert(frames->begin(), frames->end());
  auto sequential = SequentialFrameHashes(path, index, in_clip);

  // `frames`: what the scenario returns, if it is to be compared with a sequential read
  auto check = [&](const std::string &what, const HashedFramesScenario &scenario,
                   const std::vector<int> &frames = {}, bool skips_frames = false) {
    auto run = CheckSameFramesAsMappingAll(path, index, what, scenario);
    if (!is_h264) {
      EXPECT_EQ(run.result.not_decoded, 0) << path << ": " << what << ": only H.264 pictures "
                                           << "are left out";
    }
    if (!frames.empty()) {
      std::vector<size_t> expected;
      for (int frame : frames)
        expected.push_back(sequential[frame]);
      EXPECT_EQ(run.result.frames, expected) << path << ": " << what << ": not the frames that a "
                                             << "sequential read gives";
    }
    if (skips_frames && skips_non_reference) {
      EXPECT_GT(run.result.not_decoded, 0) << path << ": " << what;
      EXPECT_LT(run.result.decoded, run.expected.decoded) << path << ": " << what;
    }
  };
  check("stride 3", [&](FramesDecoderGpu &d) { return DecodeAndHash(d, stride_3); }, stride_3,
        true);
  check("a duplicate", [&](FramesDecoderGpu &d) { return DecodeAndHash(d, duplicate); },
        duplicate);
  for (auto pad : {boundary::BoundaryType::REFLECT_101, boundary::BoundaryType::CLAMP,
                   boundary::BoundaryType::CONSTANT}) {
    check(make_string("padded with ", boundary::to_string(pad)), [&](FramesDecoderGpu &d) {
      return DecodeAndHash(d, FrameRange(n - 4, n + 4), pad);
    });
  }
  check("seeks", [&](FramesDecoderGpu &d) { return ReadAndHash(d, seeks); }, seeks, true);
  check("a decode lane", [&](FramesDecoderGpu &d) {
    std::vector<size_t> frames;
    for (auto &sample : lane) {
      auto more = DecodeAndHash(d, sample);
      frames.insert(frames.end(), more.begin(), more.end());
    }
    return frames;
  }, lane_frames);
  int frame = n / 2;
  while (frame < n - 3 && index[frame].is_keyframe)
    frame++;
  check("AvSeekFrame to a frame that is not a keyframe", [&](FramesDecoderGpu &d) {
    EXPECT_TRUE(d.AvSeekFrame(index[frame].pts, frame));
    return ReadAndHash(d, FrameRange(frame, frame + 3));
  });
}

// The error of reading a frame without its data, as its picture was not decoded or as only its
// timestamp was kept (the internal errors of FramesDecoderGpu, std::logic_error)
constexpr char kPictureNotDecoded[] = "its picture was not decoded";
constexpr char kOnlyTimestampKept[] = "only its timestamp was kept";

// Checks that `e` is the error of reading a frame without its data: for this `reason`, if given
void ExpectReadWithoutData(const std::exception &e, const char *reason = nullptr) {
  std::string what = e.what();
  EXPECT_NE(what.find("Internal error: frame "), std::string::npos) << what;
  EXPECT_NE(what.find(" is read, but "), std::string::npos) << what;
  if (reason) {
    EXPECT_NE(what.find(reason), std::string::npos) << what;
  }
}

}  // namespace

// A decoder decodes the pictures that it does not return (hidden by an edit list, or skipped) only
// when other pictures refer to them (H.264 non-reference pictures are left out), and it does not
// map or convert them, nor keep a frame buffer for them (nor a place in the buffer, if they are
// never read)
TEST_F(FramesDecoderGpuTest, PicturesNotReadAreNotMapped) {
  // 188 pictures hidden by an edit list, decoded before the first frame
  auto path = SintelVideoFile(4);
  FramesDecoderGpu indexed(path);
  indexed.BuildIndex();
  auto &index = indexed.GetIndex();
  auto check = [&](const std::string &what, const std::vector<int> &frame_ids) {
    auto run = CheckSameFramesAsMappingAll(path, index, what, [&](FramesDecoderGpu &d) {
      auto frames = DecodeAndHash(d, frame_ids);
      EXPECT_GT(FramesDecoderTestAccess::ParsedPictures(d), 188);
      if (dynamic_cast<GpuDecoderMappingAll *>(&d)) {
        // one for each picture hidden by the edit list (the frames read are returned as the
        // parser displays them, unless it displays one with the frame before it)
        EXPECT_GE(d.NumFrameBuffers(), 188);
      } else {
        EXPECT_LE(d.NumFrameBuffers(), 10);
        // the hidden pictures are never read: they do not take a place in the buffer either
        EXPECT_LE(d.NumBufferSlots(), 10);
        // 86 of the 188 hidden pictures are H.264 non-reference pictures
        EXPECT_GE(FramesDecoderTestAccess::NotDecodedPictures(d), 80);
      }
      return frames;
    });
    EXPECT_LE(run.result.mapped, static_cast<int64_t>(frame_ids.size()) + kMaxMappedAhead)
        << what;
  };
  check("in a row", FrameRange(0, 32));
  // the frames skipped are not mapped either: between the frames read, and before the first one
  check("stride 3", Strided(0, 40, 3));
  check("from frame 30", {30, 31});
}

TEST_F(FramesDecoderGpuTest, SameFramesAsMappingAll) {
  auto video_dir = make_string(testing::dali_extra_path(), "/db/video");
  // H.264 without a lead-in: 13 I- and P-frames, then B-frames, of which the non-reference ones
  // (nal_ref_idc 0) are the frames 13, 16, 19, 21, 24, 26, 28, 31, 34, 37, 39 and more
  CheckSameFramesAsMappingAll(SintelVideoFile(0), true);
  // H.264 with 188 pictures hidden by an edit list, and with B-frames; HEVC
  std::vector<std::string> paths = {SintelVideoFile(4), SintelClip(2, 8),
                                    make_string(video_dir, "/vfr/test_1.mp4")};
  if (FramesDecoderGpu::SupportsHevc()) {
    paths.push_back(make_string(video_dir, "/hevc/sintel_trailer-720p.mp4"));
    paths.push_back(make_string(video_dir, "/vfr/test_2_hevc.mp4"));
  }
  // Every picture of these is decoded (see FramesDecoderGpu::ProcessPictureDecode): only the
  // mapping differs
  const std::pair<std::string, AVCodecID> without_decode_skip[] = {
      {make_string(video_dir, "/vp8/vp8.webm"), AV_CODEC_ID_VP8},
      {make_string(video_dir, "/vp9/vp9_0.mp4"), AV_CODEC_ID_VP9}};
  for (const auto &[path, codec_id] : without_decode_skip) {
    if (FramesDecoderGpu::IsCodecSupported(codec_id)) {
      paths.push_back(path);
    } else {
      GTEST_LOG_(WARNING) << "Skipping " << path << ": the codec is not supported by this GPU";
    }
  }
  for (auto &path : paths)
    CheckSameFramesAsMappingAll(path);
}

// Frames that share a pts. In this H.264 clip (the first 65 frames of SintelVideoFile(0), remuxed
// to MPEG-TS), every non-reference picture (nal_ref_idc 0) takes the pts of the frame shown right
// before it: 22 pairs of frames share one. Reading every other frame, one frame of each pair is
// read and the other one skipped. A non-reference picture decoded while the frame before it, with
// the same pts, is the next frame but only skipped must still be decoded, as it may be the frame
// read after it: otherwise reading that frame fails, as its picture was not decoded. Whether it
// is decoded at that point depends on when the parser decodes and displays the pictures. This
// checks the frames read, not which pictures are left out (see SameFramesAsMappingAll).
TEST_F(FramesDecoderGpuTest, SameFramesAsMappingAllWithSharedPts) {
  auto path = make_string(testing::dali_extra_path(),
                          "/db/video/duplicate_pts/h264_nonref_dup_pts.ts");
  FramesDecoderGpu indexed(path);
  ASSERT_TRUE(indexed.IsValid()) << path;
  indexed.BuildIndex();
  auto &index = indexed.GetIndex();
  int n = index.size();
  ASSERT_EQ(n, 65) << path;
  int shared_pts = 0;
  for (int i = 1; i < n; i++) {
    if (index[i].pts != index[i - 1].pts)
      continue;
    shared_pts++;
    // the frames of a pair are in decoding order (the reference picture first): the sort by pts
    // alone could give either order
    EXPECT_LT(index[i - 1].decode_idx, index[i].decode_idx)
        << path << ": frames " << i - 1 << " and " << i << " share a pts";
  }
  ASSERT_EQ(shared_pts, 22) << path;

  std::set<int> all_frames;
  for (int i = 0; i < n; i++)
    all_frames.insert(i);
  auto sequential = SequentialFrameHashes(path, index, all_frames);
  // the frames of a pair are different pictures: a decoder that returned the same one for both
  // would not pass for the reference
  for (int i = 1; i < n; i++) {
    if (index[i].pts == index[i - 1].pts) {
      EXPECT_NE(sequential[i], sequential[i - 1])
          << path << ": frames " << i - 1 << " and " << i << " share a pts, but not a picture";
    }
  }
  for (int start : {0, 1}) {
    auto frames = Strided(start, n, 2);
    auto what = make_string("every other frame from ", start);
    try {
      auto run = CheckSameFramesAsMappingAll(path, index, what, [&](FramesDecoderGpu &d) {
        return DecodeAndHash(d, frames);
      });
      std::vector<size_t> expected;
      for (int frame : frames)
        expected.push_back(sequential[frame]);
      EXPECT_EQ(run.result.frames, expected) << path << ": " << what << ": not the frames that a "
                                             << "sequential read gives";
    } catch (std::exception &e) {
      ADD_FAILURE() << path << ": " << what << ": " << e.what();
    }
  }
}

// Timestamps that do not fit in 32 bits (MPEG-TS has 33-bit timestamps, and a capture often starts
// above 2^31). This H.264 clip is the first 65 frames of SintelVideoFile(0), remuxed to MPEG-TS with
// every timestamp moved 30000 s later. A buffered frame is looked up by its whole timestamp: it is
// found, and the frames are the same pictures as those of the source, read in a row or with a
// stride (without a seek).
TEST_F(FramesDecoderGpuTest, TimestampsAbove32Bits) {
  auto path = make_string(testing::dali_extra_path(), "/db/video/large_pts/h264_large_pts.ts");
  FramesDecoderGpu indexed(path);
  ASSERT_TRUE(indexed.IsValid()) << path;
  indexed.BuildIndex();
  auto &index = indexed.GetIndex();
  int n = index.size();
  ASSERT_EQ(n, 65) << path;
  ASSERT_GT(index[0].pts, std::numeric_limits<int32_t>::max()) << path;

  auto source = SintelVideoFile(0);
  FramesDecoderGpu source_indexed(source);
  source_indexed.BuildIndex();
  std::set<int> all_frames;
  for (int i = 0; i < n; i++)
    all_frames.insert(i);
  auto expected = SequentialFrameHashes(source, source_indexed.GetIndex(), all_frames);
  EXPECT_EQ(SequentialFrameHashes(path, index, all_frames), expected)
      << path << ": not the frames of " << source;
  static_assert(FramesDecoderBase::kMinimumSeekLeap > 3, "the frames below are read without a seek");
  for (int stride : {2, 3}) {
    auto frames = Strided(0, n, stride);
    auto what = make_string("stride ", stride);
    auto run = CheckSameFramesAsMappingAll(path, index, what, [&](FramesDecoderGpu &d) {
      return DecodeAndHash(d, frames);
    });
    std::vector<size_t> expected_frames;
    for (int frame : frames)
      expected_frames.push_back(expected[frame]);
    EXPECT_EQ(run.result.frames, expected_frames) << path << ": " << what << ": not the frames of "
                                                  << source;
  }
}

// Reading a frame without its data is an internal error, not a frame of garbage. This decoder maps
// none of the frames that it buffers, and does not decode a non-reference picture that is decoded
// before it is the frame read (GpuDecoderMappingNone). Reading in a row, a frame is buffered when
// the parser displays it in the same cuvidParseVideoData call as the frame before it, or at the end
// of the stream; and a non-reference B-frame whose picture the parser decodes before it is the
// frame read is not decoded. Which of the two errors comes first depends on when the parser decodes
// and displays the pictures (the tests below cover each one), but one of them comes, at the end of
// the stream at the latest.
TEST_F(FramesDecoderGpuTest, ReadingFrameNotMappedFails) {
  GpuDecoderMappingNone decoder(SintelVideoFile(0));
  decoder.BuildIndex();
  DeviceBuffer<uint8_t> frame;
  frame.resize(decoder.FrameSize());
  try {
    for (int i = 0; i < decoder.NumFrames(); i++)
      decoder.ReadNextFrame(frame.data());
    FAIL() << "a frame that was not mapped was read";
  } catch (std::exception &e) {
    ExpectReadWithoutData(e);
  }
}

// At the end of the stream, the parser displays all the pictures that it still holds, in one call:
// the frame read is returned, and the frames shown after it are buffered (here, without their
// data). It holds at least the last three frames of this clip there: the last packet in decoding
// order is a B-frame shown before the last two frames, and the parser decodes it only at the end of
// the stream, as it takes a picture as complete only when the next one starts (see kPacketLag).
TEST_F(FramesDecoderGpuTest, ReadingBufferedFrameNotMappedFails) {
  GpuDecoderMappingNone decoder(SintelVideoFile(0));
  decoder.BuildIndex();
  int last = decoder.NumFrames() - 1;
  DeviceBuffer<uint8_t> frame;
  frame.resize(decoder.FrameSize());
  // seeks to the keyframe before it and skips the frames up to it (no error: they are not read)
  decoder.SeekFrame(last);
  ASSERT_EQ(decoder.NextFrameIdx(), last);
  try {
    decoder.ReadNextFrame(frame.data());
    FAIL() << "the last frame was read, although it was buffered without its data";
  } catch (std::exception &e) {
    ExpectReadWithoutData(e, kOnlyTimestampKept);
  }
}

// A non-reference picture that the parser decodes before it is the frame read is not decoded by
// GpuDecoderMappingNone: when the parser then displays it as the frame read, reading it fails.
// Unless the parser displays it in the same call as the frame before it (it is then buffered, and
// reading it fails as in ReadingBufferedFrameNotMappedFails), or decodes it only once it is the
// frame read (it is then decoded, and read fine). Which non-reference frames this happens to
// depends on when the parser decodes and displays the pictures, which a test cannot set with a
// real parser: this reads each of the non-reference B-frames below after skipping the frames
// before it, and checks that this fails so for at least one of them.
TEST_F(FramesDecoderGpuTest, ReadingFrameNotDecodedFails) {
  auto path = SintelVideoFile(0);
  FramesDecoderGpu indexed(path);
  indexed.BuildIndex();
  int not_decoded = 0, buffered = 0, read = 0;
  // the first non-reference pictures of the clip (nal_ref_idc 0), all of them with the keyframe 0
  for (int frame : {13, 16, 19, 21, 24, 26, 28, 31, 34, 37, 39}) {
    GpuDecoderMappingNone decoder(path);
    decoder.SetIndex(indexed.GetIndex());
    DeviceBuffer<uint8_t> out;
    out.resize(decoder.FrameSize());
    decoder.SeekFrame(frame);  // decodes through, skipping the frames before it
    ASSERT_EQ(decoder.NextFrameIdx(), frame);
    try {
      decoder.ReadNextFrame(out.data());
      read++;
    } catch (std::exception &e) {
      ExpectReadWithoutData(e);
      if (std::string(e.what()).find(kPictureNotDecoded) != std::string::npos)
        not_decoded++;
      else
        buffered++;
    }
  }
  EXPECT_GT(not_decoded, 0) << "no frame read failed as its picture was not decoded (" << buffered
                            << " failed as buffered without their data, " << read << " were read)";
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

// An index that is not sorted by timestamp is refused (also in a release build): the GPU decoder
// searches it by timestamp
TEST_F(FramesDecoderTest_CpuOnlyTests, SetIndexRejectsUnsortedIndex) {
  FramesDecoderCpu indexed(cfr_videos_paths_[0]);
  indexed.BuildIndex();
  FrameIndex index = indexed.GetIndex();
  ASSERT_GE(index.size(), 2u);
  FramesDecoderCpu decoder(cfr_videos_paths_[0]);
  decoder.SetIndex(index);  // a sorted index is accepted
  std::swap(index.index.front(), index.index.back());
  EXPECT_THROW(decoder.SetIndex(index), std::invalid_argument);
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

// A conversion that fails once the output surface is mapped (here: for an output type that it does
// not support) unmaps the surface. Otherwise the decoder goes back to the NVDEC cache with the
// surface still mapped, and when all of its output surfaces are taken, a decoder that gets it from
// the cache can no longer map any picture.
TEST_F(FramesDecoderGpuTest, SurfaceUnmappedAfterFailedConversion) {
  for (int i = 0; i < 8; i++) {  // more than the output surfaces of the cached decoders
    FramesDecoderGpu decoder(cfr_videos_paths_[0]);
    decoder.BuildIndex();
    decoder.SetOutputType(DALI_INT16);
    // The error of the conversion, not the generic one that the parser returns on a failed callback
    try {
      decoder.ReadNextFrame(FrameData());
      ADD_FAILURE() << "Expected the conversion to fail";
    } catch (const std::exception &e) {
      EXPECT_NE(std::string(e.what()).find("Unsupported type:"), std::string::npos) << e.what();
    }
  }
  size_t num_created = frame_dec_gpu_impl::NVDECDecodersCreated();
  FramesDecoderGpu decoder(cfr_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
  // the decoder was taken from the cache, where the failed conversions left it
  EXPECT_EQ(frame_dec_gpu_impl::NVDECDecodersCreated(), num_created);
}

// When none of the cached decoders that were set up for the format of a stream is free, a larger
// free one is reconfigured for it, instead of creating a decoder, which takes much longer (and
// destroying one, when the cache is full). It has to decode the stream as a new decoder would.
TEST_F(FramesDecoderGpuTest, DecoderCacheReconfiguresLargerDecoder) {
  using frame_dec_gpu_impl::NVDECDecodersCreated;
  using frame_dec_gpu_impl::NVDECDecodersReconfigured;
  // 800x600, coded as 800x608: the reconfigured decoder has to crop it
  std::string small_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4");
  // 1280x720
  std::string large_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4");
  auto open = [](const std::string &path) {
    auto decoder = std::make_unique<FramesDecoderGpu>(path);
    decoder->BuildIndex();
    return decoder;
  };
  auto num_set_up = [] { return NVDECDecodersCreated() + NVDECDecodersReconfigured(); };

  // Take the cached decoders that are set up for the small video, until one has to be set up for
  // it (created, or reconfigured from a larger one): then none that is set up for it is free.
  // A decoder gets its NVDEC decoder when it starts decoding.
  std::vector<std::unique_ptr<FramesDecoderGpu>> taken;
  for (size_t before = num_set_up(); num_set_up() == before;) {
    ASSERT_LT(taken.size(), 200u);
    taken.push_back(open(small_path));
    ASSERT_TRUE(taken.back()->IsValid());
    taken.back()->ReadNextFrame(FrameData());
  }

  // A larger decoder is free when this one is gone
  {
    auto large = open(large_path);
    ASSERT_TRUE(large->IsValid());
    ASSERT_EQ(large->Width(), 1280);
    large->ReadNextFrame(FrameData());
  }

  size_t num_created = NVDECDecodersCreated();
  size_t num_reconfigured = NVDECDecodersReconfigured();
  auto decoder = open(small_path);
  ASSERT_TRUE(decoder->IsValid());
  RunTest(*decoder, cfr_videos_[1]);
  EXPECT_EQ(NVDECDecodersCreated(), num_created);
  EXPECT_EQ(NVDECDecodersReconfigured(), num_reconfigured + 1);
}

// A decoder that is destroyed after a partial read goes back to the cache with pictures that NVDEC
// decoded and nobody mapped: the ones that the parser holds for the display delay, the ones that a
// read without output or a seek leaves behind. Reconfiguring it for another stream has to give the
// same frames as a new decoder does, whatever was left behind.
TEST_F(FramesDecoderGpuTest, DecoderCacheReconfigureAfterPartialReads) {
  using frame_dec_gpu_impl::NVDECDecodersCreated;
  using frame_dec_gpu_impl::NVDECDecodersReconfigured;
  using frame_dec_gpu_impl::NVDECDestroyUnusedDecoders;
  // 800x600
  std::string small_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4");
  // 1280x720, with B-frames
  std::string large_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4");
  auto open = [](const std::string &path) {
    auto decoder = std::make_unique<FramesDecoderGpu>(path);
    decoder->BuildIndex();
    return decoder;
  };

  // frames read with output, frames read without output (skipped), frame to seek to (or -1)
  struct Partial {
    int read, skipped, seek_to;
  };
  const Partial partial_reads[] = {
    {1, 0, -1}, {2, 0, -1}, {3, 0, -1}, {5, 0, -1}, {8, 0, -1}, {13, 0, -1},
    {0, 1, -1}, {0, 2, -1}, {0, 5, -1}, {1, 3, -1}, {2, 7, -1}, {0, 20, -1},
    {1, 0, 9}, {1, 0, 10}, {2, 4, 17}, {0, 3, 23}, {3, 0, 31}, {0, 1, 40},
  };
  for (int round = 0; round < 3; round++) {
    for (const auto &p : partial_reads) {
      SCOPED_TRACE(make_string("round ", round, ": read ", p.read, ", skipped ", p.skipped,
                               ", seek to ", p.seek_to));
      // Only the decoder that is created for the large video is cached, so the small video
      // reconfigures it
      NVDECDestroyUnusedDecoders();
      {
        auto large = open(large_path);
        ASSERT_TRUE(large->IsValid());
        if (p.seek_to >= 0)
          large->SeekFrame(p.seek_to);
        for (int i = 0; i < p.skipped; i++)
          large->ReadNextFrame(nullptr);
        for (int i = 0; i < p.read; i++)
          large->ReadNextFrame(FrameData());
      }
      size_t num_created = NVDECDecodersCreated();
      size_t num_reconfigured = NVDECDecodersReconfigured();
      auto decoder = open(small_path);
      ASSERT_TRUE(decoder->IsValid());
      RunTest(*decoder, cfr_videos_[1]);
      EXPECT_EQ(NVDECDecodersCreated(), num_created);
      EXPECT_EQ(NVDECDecodersReconfigured(), num_reconfigured + 1);
      if (HasFatalFailure() || HasNonfatalFailure())
        return;
    }
  }
}

// A cached decoder that is smaller than the stream cannot be reconfigured for it: a decoder is
// created instead and the small one stays in the cache. A failed reconfiguration would also end
// with a created decoder, but it would drop the small one, so the cache is checked as well.
TEST_F(FramesDecoderGpuTest, DecoderCacheDoesNotReconfigureSmallerDecoder) {
  using frame_dec_gpu_impl::NVDECDecodersCreated;
  using frame_dec_gpu_impl::NVDECDecodersReconfigured;
  using frame_dec_gpu_impl::NVDECDestroyUnusedDecoders;
  // 800x600
  std::string small_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4");
  // 1280x720
  std::string large_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4");
  auto open = [](const std::string &path) {
    auto decoder = std::make_unique<FramesDecoderGpu>(path);
    decoder->BuildIndex();
    return decoder;
  };
  auto num_set_up = [] { return NVDECDecodersCreated() + NVDECDecodersReconfigured(); };

  // Start from an empty cache: when earlier tests filled it up, creating a decoder evicts a free
  // one, which could be the small decoder checked at the end.
  NVDECDestroyUnusedDecoders();

  // Take cached decoders until one has to be created for the large video: none that is free fits
  // it any more (it would have been reconfigured), and they stay taken.
  std::vector<std::unique_ptr<FramesDecoderGpu>> taken;
  for (size_t before = NVDECDecodersCreated(); NVDECDecodersCreated() == before;) {
    ASSERT_LT(taken.size(), 200u);
    taken.push_back(open(large_path));
    ASSERT_TRUE(taken.back()->IsValid());
    taken.back()->ReadNextFrame(FrameData());
  }

  // Take the free decoders for the small video, until one has to be set up for it. That one is
  // the only free decoder when it is put back: it is smaller than the large video.
  for (size_t before = num_set_up(); num_set_up() == before;) {
    ASSERT_LT(taken.size(), 200u);
    taken.push_back(open(small_path));
    ASSERT_TRUE(taken.back()->IsValid());
    taken.back()->ReadNextFrame(FrameData());
  }
  taken.pop_back();

  size_t num_created = NVDECDecodersCreated();
  size_t num_reconfigured = NVDECDecodersReconfigured();
  {
    auto large = open(large_path);
    ASSERT_TRUE(large->IsValid());
    large->ReadNextFrame(FrameData());
    EXPECT_EQ(NVDECDecodersCreated(), num_created + 1);
    EXPECT_EQ(NVDECDecodersReconfigured(), num_reconfigured);
  }

  // The small decoder was not touched: it is reused as it is, and not reconfigured from the new
  // large one that was just put back.
  num_created = NVDECDecodersCreated();
  num_reconfigured = NVDECDecodersReconfigured();
  auto small = open(small_path);
  ASSERT_TRUE(small->IsValid());
  small->ReadNextFrame(FrameData());
  EXPECT_EQ(NVDECDecodersCreated(), num_created);
  EXPECT_EQ(NVDECDecodersReconfigured(), num_reconfigured);
}

TEST_F(FramesDecoderGpuTest, IndexerGivesTheIndexOfTheDecoder) {
  std::vector<std::string> paths;
  for (const auto *list : {&cfr_videos_paths_, &vfr_videos_paths_, &cfr_mpeg4_videos_paths_,
                           &vfr_mpeg4_videos_paths_}) {
    paths.insert(paths.end(), list->begin(), list->end());
  }
  if (FramesDecoderGpu::SupportsHevc()) {
    paths.insert(paths.end(), cfr_hevc_videos_paths_.begin(), cfr_hevc_videos_paths_.end());
    paths.insert(paths.end(), vfr_hevc_videos_paths_.begin(), vfr_hevc_videos_paths_.end());
  }
  // Clips with packets hidden by an edit list, where BuildIndex asks the demuxer where seeks land
  // (the reader caches the index built by FramesIndexer, so seek_decode_idx must match too)
  paths.push_back(SintelClip(0, 9));
  paths.push_back(SintelClip(2, 8));
  ASSERT_FALSE(paths.empty());
  for (const auto &path : paths) {
    SCOPED_TRACE(path);
    FramesDecoderGpu decoder(path);
    FramesIndexer indexer(path);
    ExpectIndexerGivesTheIndexOfTheDecoder(decoder, indexer);
  }
}

// The files that the video reader skips as invalid are the same with the indexer as with the
// decoder: a missing file, a file without a video stream, and a codec that is not supported by the
// GPU decoder. The indexer only reports the codec, the reader checks it with the decoder.
TEST_F(FramesDecoderGpuTest, IndexerIsValidForTheSameFilesAsTheDecoder) {
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  for (const auto &path : {make_string(video_dir, "/nonexistent.mp4"),
                           make_string(video_dir, "/corrupted/audio_only.mp4"),
                           make_string(video_dir, "/mjpeg/mjpeg.avi"), cfr_videos_paths_[0]}) {
    SCOPED_TRACE(path);
    FramesDecoderGpu decoder(path);
    FramesIndexer indexer(path);
    bool indexer_valid =
        indexer.IsValid() && FramesDecoderGpu::IsStreamSupported(indexer.CodecParameters());
    EXPECT_EQ(indexer_valid, decoder.IsValid());
  }
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(cfr_hevc_videos_paths_[0]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

// After a seek to an HEVC CRA picture, the parser drops the CRA's leading pictures, which reference
// pictures before the seek point, and the timestamps that it then reports with the pictures it
// displays are not those of their packets. The frames read after seeks must be the frames that a
// sequential read gives.
TEST_F(FramesDecoderGpuTest, HevcSeekToOpenGopKeyframes) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  // a keyframe every few frames, open GOPs (CRA pictures with leading pictures), B-frames
  auto path = make_string(testing::dali_extra_path(), "/db/video/hevc/sintel_trailer-720p.mp4");
  std::vector<int> frames = {64, 70, 192, 384, 396, 480, 704, 705, 710, 1024, 1040, 1184, 1200};
  FramesDecoderGpu sequential(path);
  sequential.BuildIndex();
  ASSERT_GT(sequential.NumFrames(), frames.back());
  size_t frame_size = sequential.FrameSize();
  DeviceBuffer<uint8_t> frame_gpu;
  frame_gpu.resize(frame_size);
  std::vector<std::vector<uint8_t>> expected;
  std::vector<uint8_t> actual(frame_size);
  for (int i = 0, next = 0; next < static_cast<int>(frames.size()); i++) {
    ASSERT_TRUE(sequential.ReadNextFrame(frame_gpu.data()));
    if (i == frames[next]) {
      expected.emplace_back(frame_size);
      MemCopy(expected.back().data(), frame_gpu.data(), frame_size);
      next++;
    }
  }

  FramesDecoderGpu decoder(path);
  decoder.BuildIndex();
  std::vector<int> order(frames.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937(0));
  for (int k : order) {
    decoder.SeekFrame(frames[k]);
    ASSERT_EQ(decoder.NextFrameIdx(), frames[k]);
    ASSERT_TRUE(decoder.ReadNextFrame(frame_gpu.data()));
    MemCopy(actual.data(), frame_gpu.data(), frame_size);
    EXPECT_EQ(actual, expected[k]) << "frame " << frames[k];
  }
}

TEST_F(FramesDecoderGpuTest, VariableFrameRateHevc) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(vfr_hevc_videos_paths_[1]);
  decoder.BuildIndex();
  RunTest(decoder, vfr_hevc_videos_[1]);
}

// With these codecs the parser decodes a picture while it parses its packet, and with the display
// delay the last picture is displayed only at the end of the stream: reading it, sequentially or
// after a seek, must not fail.
TEST_F(FramesDecoderGpuTest, ReadsTheLastFrameOfCodecsWithoutDecodeLag) {
  // GPU and CPU frames of these clips differ by 3e-5 on average, neighbouring frames by 0.07 at
  // least and an all-zeros frame by 85: this tolerates the former and catches the others.
  constexpr double kMaxAvgError = 0.02;
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  const std::pair<std::string, AVCodecID> videos[] = {
      {make_string(video_dir, "/vp8/vp8.webm"), AV_CODEC_ID_VP8},
      {make_string(video_dir, "/vp9/vp9_0.mp4"), AV_CODEC_ID_VP9}};
  int num_tested = 0;
  for (const auto &[path, codec_id] : videos) {
    SCOPED_TRACE(path);
    if (!FramesDecoderGpu::IsCodecSupported(codec_id)) {
      GTEST_LOG_(WARNING) << "Skipping " << path << ": the codec is not supported by this GPU";
      continue;
    }
    num_tested++;
    FramesDecoderGpu decoder(path);
    ASSERT_TRUE(decoder.IsValid());
    decoder.BuildIndex();
    // The reference is the CPU decoder, which decodes these codecs independently of NVDEC.
    FramesDecoderCpu reference(path);
    ASSERT_TRUE(reference.IsValid());
    int num_frames = decoder.NumFrames();
    ASSERT_GT(num_frames, 4);
    ASSERT_EQ(reference.NumFrames(), num_frames);
    size_t frame_size = decoder.FrameSize();
    ASSERT_EQ(reference.FrameSize(), frame_size);
    DeviceBuffer<uint8_t> frame_gpu;
    frame_gpu.resize(frame_size);
    std::vector<uint8_t> actual(frame_size), expected(frame_size);

    // Reads the next frame with both decoders and compares them. The two do the color conversion
    // differently, so the frames are close, not equal. An all-zeros or an unwritten frame, or the
    // wrong frame (the frames have their number drawn on them), is far from that.
    auto read_and_compare = [&](int i) {
      ASSERT_EQ(decoder.NextFrameIdx(), i);
      ASSERT_EQ(reference.NextFrameIdx(), i);
      CUDA_CALL(cudaMemset(frame_gpu.data(), 0, frame_size));
      ASSERT_TRUE(decoder.ReadNextFrame(frame_gpu.data()));
      ASSERT_TRUE(reference.ReadNextFrame(expected.data()));
      MemCopy(actual.data(), frame_gpu.data(), frame_size);
      double sum = 0;
      for (size_t j = 0; j < frame_size; j++)
        sum += std::abs(static_cast<int>(actual[j]) - static_cast<int>(expected[j]));
      EXPECT_LT(sum / frame_size, kMaxAvgError) << "frame " << i;
    };

    for (int i = 0; i < num_frames; i++) {
      ASSERT_NO_FATAL_FAILURE(read_and_compare(i));
    }
    EXPECT_FALSE(decoder.ReadNextFrame(frame_gpu.data()));
    EXPECT_FALSE(reference.ReadNextFrame(expected.data()));

    for (int start : {num_frames - 3, 0, num_frames - 1}) {
      decoder.SeekFrame(start);
      reference.SeekFrame(start);
      for (int i = start; i < num_frames; i++) {
        ASSERT_NO_FATAL_FAILURE(read_and_compare(i));
      }
    }
  }
  if (num_tested == 0) {
    GTEST_SKIP() << "Neither VP8 nor VP9 is supported by this GPU";
  }
}

TEST_F(FramesDecoderTest_CpuOnlyTests, IndexerGivesTheIndexOfTheDecoder) {
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  for (const auto &path : {make_string(video_dir, "/vp8/vp8.webm"),
                           make_string(video_dir, "/vp9/vp9_0.mp4"),
                           make_string(video_dir, "/mjpeg/mjpeg.avi")}) {
    SCOPED_TRACE(path);
    FramesDecoderCpu decoder(path);
    FramesIndexer indexer(path);
    ExpectIndexerGivesTheIndexOfTheDecoder(decoder, indexer);
  }
}

// In MPEG-4 Part 2 AVI, the keyframes are the packets that start with a coded I-VOP, and the
// B-VOPs right after one are its leading pictures. Neither file below has leading pictures: they
// are in the open GOPs that qa/TL0_videoreader_test generates (see test_video_reader.py).
TEST_F(FramesDecoderTest_CpuOnlyTests, IndexOfMpeg4Avi) {
  struct Expected {
    std::string path;
    int frames;
    std::vector<int> keyframes;
  };
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  // From the AVI chunks: the packets of packed_bframes_test.avi are I, then P+B, B, a not-coded
  // VOP (typed P), ... ; those of test_1.avi are I-VOPs and P-VOPs only
  for (const auto &expected :
       {Expected{make_string(video_dir, "/ucf101_test/packed_bframes_test.avi"), 91, {0}},
        Expected{make_string(video_dir, "/cfr/test_1.avi"), 50, {0, 12, 24, 36, 48}}}) {
    SCOPED_TRACE(expected.path);
    FramesIndexer indexer(expected.path);
    indexer.BuildIndex();
    const auto &index = indexer.GetIndex();
    EXPECT_TRUE(index.decode_order_timestamps);
    ASSERT_EQ(index.size(), static_cast<size_t>(expected.frames));
    std::vector<int> keyframes;
    for (int i = 0; i < expected.frames; i++) {
      if (index[i].is_keyframe) {
        keyframes.push_back(i);
        EXPECT_EQ(index[i].leading_pictures, 0) << "frame " << i;
      } else {
        EXPECT_EQ(index[i].leading_pictures, -1) << "frame " << i;
      }
    }
    EXPECT_EQ(keyframes, expected.keyframes);
  }
}

// An IDR picture has no leading pictures; an HEVC CRA picture may have
TEST_F(FramesDecoderTest_CpuOnlyTests, LeadingPicturesOfIdrKeyframes) {
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  {
    FramesIndexer indexer(make_string(video_dir, "/cfr/test_1.mp4"));  // H.264
    indexer.BuildIndex();
    const auto &index = indexer.GetIndex();
    int keyframes = 0;
    for (size_t i = 0; i < index.size(); i++) {
      if (index[i].is_keyframe) {
        keyframes++;
        EXPECT_EQ(index[i].leading_pictures, 0) << "frame " << i;
      }
    }
    EXPECT_GT(keyframes, 1);
  }
  {
    // keyframes every few frames, open GOPs (CRA pictures)
    FramesIndexer indexer(make_string(video_dir, "/hevc/sintel_trailer-720p.mp4"));
    indexer.BuildIndex();
    const auto &index = indexer.GetIndex();
    int unknown = 0;
    for (size_t i = 0; i < index.size(); i++)
      unknown += index[i].is_keyframe && index[i].leading_pictures == -1;
    EXPECT_GT(unknown, 0);
  }
}

// The files that the video reader skips as invalid are the same with the indexer as with the
// decoder: a missing file, a file without a video stream, and a codec that is not supported by the
// CPU decoder. The indexer only reports the codec, the reader checks it with the decoder.
TEST_F(FramesDecoderTest_CpuOnlyTests, IndexerIsValidForTheSameFilesAsTheDecoder) {
  const std::string video_dir = make_string(testing::dali_extra_path(), "/db/video");
  for (const auto &path : {make_string(video_dir, "/nonexistent.mp4"),
                           make_string(video_dir, "/corrupted/audio_only.mp4"),
                           cfr_mpeg4_videos_paths_[0], make_string(video_dir, "/vp9/vp9_0.mp4")}) {
    SCOPED_TRACE(path);
    FramesDecoderCpu decoder(path);
    FramesIndexer indexer(path);
    bool indexer_valid =
        indexer.IsValid() && FramesDecoderCpu::IsStreamSupported(indexer.CodecParameters());
    EXPECT_EQ(indexer_valid, decoder.IsValid());
  }
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

TEST_F(FramesDecoderTest_CpuOnlyTests, InputStreamCfrVideo) {
  FramesDecoderCpu decoder(FileStream::Open(cfr_videos_paths_[1]), cfr_videos_paths_[1]);
  ASSERT_TRUE(decoder.IsValid());
  EXPECT_EQ(decoder.Filename(), cfr_videos_paths_[1]);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[1]);
}

TEST_F(FramesDecoderGpuTest, InputStreamCfrVideo) {
  FramesDecoderGpu decoder(FileStream::Open(cfr_videos_paths_[0]), cfr_videos_paths_[0]);
  ASSERT_TRUE(decoder.IsValid());
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InputStreamVfrVideoNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderCpu decoder(
      std::make_unique<MemInputStream>(memory_video.data(), memory_video.size()));
  ASSERT_TRUE(decoder.IsValid());
  RunTest(decoder, vfr_videos_[0], false);
}

TEST_F(FramesDecoderGpuTest, InputStreamVfrVideoNoIndex) {
  auto memory_video = MemoryVideo(vfr_videos_paths_[0]);
  FramesDecoderGpu decoder(
      std::make_unique<MemInputStream>(memory_video.data(), memory_video.size()));
  ASSERT_TRUE(decoder.IsValid());
  RunTest(decoder, vfr_videos_[0], false);
}

/**
 * @brief A stream whose next read fails on request, like a transient error of remote storage.
 */
class FailingInputStream : public MemInputStream {
 public:
  using MemInputStream::MemInputStream;

  enum class Failure { None, Throw, ReadNothing };

  void FailNextRead(Failure failure) {
    failure_ = failure;
  }

  size_t Read(void *buf, size_t bytes) override {
    auto failure = std::exchange(failure_, Failure::None);
    if (failure == Failure::Throw)
      throw std::runtime_error("Simulated read failure.");
    if (failure == Failure::ReadNothing)
      return 0;
    return MemInputStream::Read(buf, bytes);
  }

 private:
  Failure failure_ = Failure::None;
};

TEST_F(FramesDecoderTest_CpuOnlyTests, InputStreamReadErrorRejectsIndex) {
  // Larger than the decoder's I/O buffer, so building the index has to read past what was read
  // while opening the file.
  auto memory_video = MemoryVideo(testing::dali_extra_path() + "/db/video/mjpeg/mjpeg.avi");
  for (auto failure : {FailingInputStream::Failure::Throw,
                       FailingInputStream::Failure::ReadNothing}) {
    auto stream = std::make_unique<FailingInputStream>(memory_video.data(), memory_video.size());
    auto *failing_stream = stream.get();
    FramesDecoderCpu decoder(std::move(stream));
    ASSERT_TRUE(decoder.IsValid());
    failing_stream->FailNextRead(failure);
    // The failure is transient, so reopening the stream when the scan ends succeeds - a partial
    // index would then be taken for the whole video, and cached by the reader.
    EXPECT_THROW(decoder.BuildIndex(), std::runtime_error);
    EXPECT_FALSE(decoder.HasIndex());
  }
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

namespace {

FrameIndex MakeFrameIndex(const std::vector<int64_t> &pts) {
  FrameIndex index = {};
  index.timebase = AVRational{1, 1000};
  for (auto p : pts)
    index.index.push_back(IndexEntry{p, 0, false, false});
  return index;
}

double AvgAbsDiff(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
  int64_t sum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
  }
  return static_cast<double>(sum) / a.size();
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

TEST(FramesDecoderFrameIndexTest, EndPtsWithoutRecordedStreamEnd) {
  auto index = MakeFrameIndex({100, 200, 300});
  EXPECT_EQ(index.EndPts(), 400);  // last pts + last inter-frame gap
  EXPECT_EQ(MakeFrameIndex({5}).EndPts(), 6);
}

TEST(FramesDecoderFrameIndexTest, EndPtsUsesRecordedStreamEnd) {
  // The last frame lasts 100 ticks, the one before it 200: the gap would give 500.
  auto index = MakeFrameIndex({0, 100, 300});
  EXPECT_EQ(index.EndPts(), 500);
  index.end_pts = 400;
  EXPECT_EQ(index.EndPts(), 400);
  EXPECT_EQ(index.GetFrameIdxByTimestamp(400, true), 3);   // end of stream: one past the last
  EXPECT_EQ(index.GetFrameIdxByTimestamp(399, true), 2);   // inside the last frame
  // Single frame lasting 40 ticks: ranges starting inside it still select it.
  auto single = MakeFrameIndex({0});
  single.end_pts = 40;
  EXPECT_EQ(single.EndPts(), 40);
  EXPECT_EQ(single.GetFrameIdxByTimestamp(39, true), 0);
  EXPECT_EQ(single.GetFrameIdxByTimestamp(40, true), 1);
  // A recorded end not past the last pts is ignored.
  single.end_pts = 0;
  EXPECT_EQ(single.EndPts(), 1);
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

TEST(VideoUtilsSecondsToTimestampTest, RoundUpGoesPastTheFrameBelow) {
  // 24 fps at timebase 1/12288: 512 ticks per frame, so frame 22 has pts 22 * 512 = 11264. A time
  // a fraction of a tick above that pts must round up to 11265, so that the lookup does not take
  // it for frame 22's own pts, and exclude frame 22 from a range that ends there.
  AVRational timebase{1, 12288};
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.9167271357878531, /*round_down=*/false), 11265);
  // 0.91670 s is 11264.4096 ticks: closer to 11264 than to 11265, but still above frame 22's pts
  EXPECT_EQ(SecondsToTimestamp(timebase, 0.91670, /*round_down=*/false), 11265);
}

TEST(VideoUtilsSecondsToTimestampTest, RoundDownIsTheFloor) {
  AVRational timebase{1, 12288};
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

TEST(VideoUtilsSecondsToTimestampTest, SnapsFloatingPointNoiseAtExactTickBoundary) {
  // At timebase 1/1000 (the Matroska/WebM default), 1.001 * 1000 is not exactly 1001.0 in double
  // precision: it is 1000.9999999999999. Without snapping, floor() would return 1000 instead of
  // 1001, and a file_list boundary on that frame's pts would select the wrong frame.
  AVRational timebase{1, 1000};
  ASSERT_NE(1.001 * timebase.den / timebase.num, 1001.0)
      << "this test is meant to exercise floating-point noise; if this starts failing because "
         "1.001 * 1000 became exact, the epsilon-snapping logic is no longer being exercised";
  EXPECT_EQ(SecondsToTimestamp(timebase, 1.001, /*round_down=*/true), 1001);
  EXPECT_EQ(SecondsToTimestamp(timebase, 1.001, /*round_down=*/false), 1001);
}

TEST(VideoUtilsSecondsToTimestampTest, GenuineFractionalOffsetIsNotSnapped) {
  // A real request landing a tenth of a tick off a boundary must NOT be snapped to the
  // boundary -- only floating-point representation noise (many orders of magnitude smaller)
  // should be. 1.0001 s at timebase 1/1000 is 1000.1 ticks: 0.1 of a tick off of 1000, which is
  // far larger than the relative epsilon used for noise snapping.
  AVRational timebase{1, 1000};
  EXPECT_EQ(SecondsToTimestamp(timebase, 1.0001, /*round_down=*/true), 1000);
  EXPECT_EQ(SecondsToTimestamp(timebase, 1.0001, /*round_down=*/false), 1001);
}

TEST(VideoUtilsSecondsToTimestampTest, ExactIntegerTicksRoundTripAcrossCommonTimebases) {
  // For several common timebases, every millisecond value from 0 to 2 seconds gives exactly the
  // integer tick count, when it is one, whatever the floating-point noise in `seconds * den / num`.
  const AVRational timebases[] = {{1, 1000}, {1, 12288}, {1, 90000}, {1, 25}};
  for (const auto &timebase : timebases) {
    for (int ms = 0; ms <= 2000; ms++) {
      // Exact tick count for this millisecond value at this timebase, computed with integer
      // arithmetic (no floating point) so it's unambiguously correct.
      int64_t num = static_cast<int64_t>(ms) * timebase.den;
      int64_t den = static_cast<int64_t>(1000) * timebase.num;
      if (num % den != 0)
        continue;  // not a whole number of ticks: see GenuineFractionalOffsetIsNotSnapped
      int64_t expected_ticks = num / den;
      double seconds = ms / 1000.0;
      EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/true), expected_ticks)
          << "timebase=1/" << timebase.den << " ms=" << ms;
      EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/false), expected_ticks)
          << "timebase=1/" << timebase.den << " ms=" << ms;
TEST_F(FramesDecoderGpuTest, MjpegRejectsFramesOfOtherSizeThanDeclared) {
  if (!FramesDecoderGpu::CanNvdecDecode(cudaVideoCodec_JPEG, cudaVideoChromaFormat_444)) {
    GTEST_SKIP() << "NVDEC does not support 8-bit 4:4:4 JPEG on this GPU";
  }
  auto memory_video = MemoryVideo(
      make_string(testing::dali_extra_path(), "/db/video/mjpeg/mjpeg.avi"));
  // Declare a size that differs from the one of the frames, in the stream format chunk of the AVI
  // ('strf', the chunk size, the bitmap header size, the width and the height).
  const std::string strf = "strf";
  auto it = std::search(memory_video.begin(), memory_video.end(), strf.begin(), strf.end());
  ASSERT_NE(it, memory_video.end());
  int32_t *size = reinterpret_cast<int32_t *>(&*it + 12);
  ASSERT_EQ(size[0], 1280);
  ASSERT_EQ(size[1], 720);
  size[0] = 640;
  size[1] = 360;

  FramesDecoderCpu cpu_decoder(memory_video.data(), memory_video.size());
  ASSERT_TRUE(cpu_decoder.IsValid());
  std::vector<uint8_t> frame(cpu_decoder.FrameSize());
  EXPECT_THROW(cpu_decoder.ReadNextFrame(frame.data()), std::runtime_error);

  FramesDecoderGpu gpu_decoder(memory_video.data(), memory_video.size());
  ASSERT_TRUE(gpu_decoder.IsValid());
  DeviceBuffer<uint8_t> frame_device;
  frame_device.resize(gpu_decoder.FrameSize());
  EXPECT_THROW(gpu_decoder.ReadNextFrame(frame_device.data()), std::runtime_error);
}

TEST_F(FramesDecoderGpuTest, MjpegDecodesMultipleFrames) {
  // DALI-4918: for MJPEG the driver can report a minimum of 1 decode surface, with which decoding
  // the 2nd frame used to fail with CUDA_ERROR_INVALID_VALUE.
  //
  // Every frame is also compared against the CPU backend: NVDEC writes JPEG pictures in stream
  // order with the legacy default stream, so a color conversion on a non-blocking stream used to
  // read surfaces that weren't written yet and return blank frames (mostly the first ~20 frames
  // of a freshly created NVDEC decoder). Hence the non-blocking streams, and several decoders
  // alive at once: each one holds its NVDEC decoder, so the others can't reuse it from
  // NVDECCache and get freshly created ones.

  // The test file is 8-bit 4:4:4 (yuvj444p), which not every NVDEC generation can decode.
  if (!FramesDecoderGpu::CanNvdecDecode(cudaVideoCodec_JPEG, cudaVideoChromaFormat_444)) {
    GTEST_SKIP() << "NVDEC does not support 8-bit 4:4:4 JPEG on this GPU";
  }
  auto mjpeg_path = make_string(testing::dali_extra_path(), "/db/video/mjpeg/mjpeg.avi");
  FramesDecoderCpu cpu_decoder(mjpeg_path, DALI_RGB);
  ASSERT_TRUE(cpu_decoder.IsValid());
  cpu_decoder.BuildIndex();
  int num_frames = cpu_decoder.NumFrames();
  ASSERT_GT(num_frames, 1);

  constexpr int kNumGpuDecoders = 3;
  std::vector<CUDAStream> streams;
  std::vector<std::unique_ptr<FramesDecoderGpu>> gpu_decoders;
  for (int d = 0; d < kNumGpuDecoders; ++d) {
    streams.push_back(CUDAStream::Create(true));
    gpu_decoders.push_back(std::make_unique<FramesDecoderGpu>(mjpeg_path, streams[d], DALI_RGB));
    ASSERT_TRUE(gpu_decoders[d]->IsValid());
    gpu_decoders[d]->BuildIndex();
    ASSERT_EQ(gpu_decoders[d]->NumFrames(), num_frames);
    ASSERT_EQ(gpu_decoders[d]->FrameSize(), cpu_decoder.FrameSize());
  }

  std::vector<uint8_t> expected(cpu_decoder.FrameSize());
  std::vector<uint8_t> actual(cpu_decoder.FrameSize());
  DeviceBuffer<uint8_t> frame_device;
  frame_device.resize(cpu_decoder.FrameSize());
  for (int i = 0; i < num_frames; ++i) {
    ASSERT_TRUE(cpu_decoder.ReadNextFrame(expected.data())) << "CPU: failed decoding frame " << i;
    for (int d = 0; d < kNumGpuDecoders; ++d) {
      ASSERT_TRUE(gpu_decoders[d]->ReadNextFrame(frame_device.data()))
          << "GPU decoder " << d << ": failed decoding frame " << i;
      CUDA_CALL(cudaMemcpyAsync(actual.data(), frame_device.data(), actual.size(),
                                cudaMemcpyDeviceToHost, streams[d]));
      CUDA_CALL(cudaStreamSynchronize(streams[d]));
      // The two backends use different JPEG decoders and color conversions, so they differ by
      // rounding noise (below 1 on average); a blank frame is off by tens.
      double diff = AvgAbsDiff(actual, expected);
      EXPECT_LT(diff, 1.5) << "GPU decoder " << d << ": frame " << i
                           << " differs from the CPU backend's (avg abs diff " << diff << ")";
    }
  }
}

TEST(VideoUtilsSecondsToTimestampTest, LargeTickCountsStayDirectional) {
  // The noise-snapping tolerance must stay far below half a tick for long videos at fine
  // timebases, or every value would snap to the nearest tick instead of being floored/ceiled.
  AVRational timebase{1, 90000};
  double seconds = 900000.3 / 90000;  // ~900000.3 ticks (10 s)
  EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/false), 900001);
  EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/true), 900000);
  seconds = 3240000000.3 / 90000;  // ~3240000000.3 ticks (10 h)
  EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/false), 3240000001);
  EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/true), 3240000000);
}

TEST(VideoUtilsSecondsToTimestampTest, SnapsFloatingPointNoiseAtAnyMagnitude) {
  // k / 30 s at timebase 1/30 is exactly k ticks, but the double product isn't always exactly k.
  AVRational timebase{1, 30};
  for (int64_t k : {123, 500005}) {
    double seconds = k / 30.0;
    ASSERT_NE(seconds * timebase.den / timebase.num, static_cast<double>(k))
        << "meant to exercise floating-point noise at k=" << k;
    EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/true), k);
    EXPECT_EQ(SecondsToTimestamp(timebase, seconds, /*round_down=*/false), k);
  }
  // NTSC timebase: 1001/30000 s is exactly 1 tick, computed as 1.0000000000000002.
  AVRational ntsc{1001, 30000};
  double one_tick = 1001.0 / 30000;
  ASSERT_NE(one_tick * ntsc.den / ntsc.num, 1.0);
  EXPECT_EQ(SecondsToTimestamp(ntsc, one_tick, /*round_down=*/false), 1);
  EXPECT_EQ(SecondsToTimestamp(ntsc, one_tick, /*round_down=*/true), 1);
}

TEST(VideoUtilsSecondsToTimestampTest, DecimalTimesGiveExactTicks) {
  // At timebase 1/90000, 200.1 s is 18009000 ticks, 128.04 s is 11523600 ticks, and every tenth
  // of a second is a multiple of 9000 ticks: rounding down or up must give exactly that.
  AVRational timebase{1, 90000};
  for (bool round_down : {true, false}) {
    EXPECT_EQ(SecondsToTimestamp(timebase, std::strtod("200.1", nullptr), round_down), 18009000);
    EXPECT_EQ(SecondsToTimestamp(timebase, std::strtod("128.04", nullptr), round_down), 11523600);
  }
  int wrong = 0;
  for (int k = 1; k <= 36000; k++) {  // up to one hour
    double seconds = std::strtod(make_string(k / 10, ".", k % 10).c_str(), nullptr);
    for (bool round_down : {true, false}) {
      int64_t ticks = SecondsToTimestamp(timebase, seconds, round_down);
      if (ticks != int64_t(k) * 9000 && wrong++ == 0)
        ADD_FAILURE() << k / 10 << "." << k % 10 << " s, round down " << round_down << ": "
                      << ticks;
    }
  }
  EXPECT_EQ(wrong, 0);
}

TEST(VideoUtilsGetVideoFilesTest, FileListTimesAreParsedAsDoubles) {
  // As floats, times from 128 s on can be a tick off at timebase 1/90000, once rounded.
  char path[] = "/tmp/file_list_times_XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  close(fd);
  auto remove_list = AtScopeExit([&] { std::remove(path); });
  {
    std::ofstream list(path);
    // float(200.1) != 200.1, and 16777217 (2^24 + 1) is the first integer a float cannot hold
    list << "video.mp4 0 200.1 300.25\nvideo.mp4 1 16777217 16777219\n";
  }
  auto files = GetVideoFiles("", {}, false, {}, path);
  ASSERT_EQ(files.size(), 2u);
  EXPECT_EQ(files[0].start, 200.1);
  EXPECT_EQ(files[0].end, 300.25);
  EXPECT_EQ(files[1].start, 16777217.0);
  EXPECT_EQ(files[1].end, 16777219.0);
}

TEST(VideoUtilsSecondsToTimestampTest, SaturatesOutOfRangeTimes) {
  AVRational timebase{1, 90000};
  const int64_t kMax = int64_t(1) << 62;
  for (bool round_down : {true, false}) {
    EXPECT_EQ(SecondsToTimestamp(timebase, 1e20, round_down), kMax);
    EXPECT_EQ(SecondsToTimestamp(timebase, -1e20, round_down), -kMax);
    EXPECT_EQ(SecondsToTimestamp(timebase, std::numeric_limits<double>::infinity(), round_down),
              kMax);
    EXPECT_EQ(SecondsToTimestamp(timebase, -std::numeric_limits<double>::infinity(), round_down),
              -kMax);
    EXPECT_THROW(
        SecondsToTimestamp(timebase, std::numeric_limits<double>::quiet_NaN(), round_down),
        std::invalid_argument);
  }
  // Leaves headroom for adding a pts offset, so a huge start is still "past the end".
  EXPECT_GT(SecondsToTimestamp(timebase, 1e20, true) + (int64_t(1) << 61), 0);
}

}  // namespace dali
