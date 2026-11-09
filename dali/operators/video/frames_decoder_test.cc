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
#include <cmath>
#include <cstdlib>
#include <exception>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/cuda_error.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/video_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/file.h"

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
    EXPECT_EQ(actual.index[i].seek_timestamp, expected.index[i].seek_timestamp) << "frame " << i;
  }
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

// The timestamps (pts) of the packets where seeks to `timestamps` land, with av_seek_frame as
// AvSeekFrame seeks, asked to a demuxer of its own
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
  int first_packet = 0;  // the frame of the packet at decoding position 0
  while (index[first_packet].decode_idx != 0)
    first_packet++;
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
  std::vector<int64_t> timestamps, expected;
  for (int target : targets) {
    timestamps.push_back(index.SeekTimestamp(target));
    expected.push_back(index[target].is_keyframe ? index[target].pts : index[first_packet].pts);
  }
  auto landings = SeekLandings(path, timestamps);
  ASSERT_EQ(landings.size(), timestamps.size());
  for (size_t i = 0; i < landings.size(); i++) {
    EXPECT_EQ(landings[i], expected[i]) << path << ": a seek to " << timestamps[i]
                                        << " is to land on the packet with the pts " << expected[i];
  }
}

std::string SintelVideoFile(int clip) {
  return make_string(testing::dali_extra_path(), "/db/video/sintel/video_files/",
                     "sintel_trailer-720p_", clip, ".mp4");
}

// An MPEG-TS copy of a clip and the MP4 clip it was copied from: the same packets (in Annex B) with
// their timestamps moved, and the same frames
struct MpegTsClip {
  std::string ts, mp4;
  bool hevc;
  int keyframes;  // the pictures that decoding can start from
};

std::vector<MpegTsClip> MpegTsClips() {
  auto dir = make_string(testing::dali_extra_path(), "/db/video/");
  return {{make_string(dir, "mpegts/test_1.ts"), make_string(dir, "cfr/test_1.mp4"), false, 6},
          {make_string(dir, "mpegts/test_2_hevc.ts"), make_string(dir, "cfr/test_2_hevc.mp4"), true,
           2},
          // only the first picture is an IDR picture, but the MPEG-TS demuxer also flags the
          // I-pictures with a recovery point (open GOPs) as keyframes
          {make_string(dir, "mpegts/open_gop_h264.ts"),
           make_string(dir, "non_idr_keyframes/open_gop_h264.mp4"), false, 1}};
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
 * decodes the pictures that AddDecodeWork estimates from `pos`, reopens the file as estimated,
 * and ends where `pos` does.
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

  int64_t before = decoder.DecodedPictures();
  int64_t reopens_before = decoder.Reopens();
  decoder.DecodeFrames(buffer, make_cspan(frame_ids), boundary::BoundaryType::CLAMP);
  int64_t ahead = decoder.DecodedPictures() - before - work.pictures;
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
 * pictures decoded for each frame: in total, the decoder is never behind the estimate and at most
 * the pictures of the next `lookahead` frames ahead of it (see kPacketLag). Returns the number of
 * frames that cost nothing (decoded earlier, for a frame shown before them).
 */
int CheckPositionAt(FramesDecoderBase &decoder, uint8_t *frame, int lookahead,
                    const std::string &path) {
  auto &index = decoder.GetIndex();
  int free_frames = 0;
  int64_t start = decoder.DecodedPictures(), estimate = 0;
  for (int k = 0; k < static_cast<int>(index.size()); k++) {
    auto pos = FramesDecoderBase::PositionAt(index, k);
    FramesDecoderBase::DecodeWork work;
    FramesDecoderBase::AddDecodeWork(index, pos, k, work);
    estimate += work.pictures;
    EXPECT_EQ(decoder.NextFrameIdx(), k) << path;
    decoder.ReadNextFrame(frame);
    int64_t ahead = decoder.DecodedPictures() - start - estimate;
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
  // MPEG-TS, whose demuxer seeks by decoding timestamp
  for (auto &clip : MpegTsClips()) {
    if (!clip.hevc || FramesDecoderGpu::SupportsHevc())
      paths.push_back(clip.ts);
  }
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

// MPEG-TS stores H.264 and HEVC in Annex B (NAL units after start codes, not after their length),
// and its demuxer seeks by decoding timestamp, on any packet: the index of an MPEG-TS copy has the
// keyframes (also where the demuxer flags other pictures as keyframes) and the decoding order of the
// MP4 clip, and a seek to each keyframe lands on it
TEST_F(FramesDecoderGpuTest, MpegTsIndex) {
  for (auto &clip : MpegTsClips()) {
    FramesIndexer ts_indexer(clip.ts), mp4_indexer(clip.mp4);
    ASSERT_TRUE(ts_indexer.IsValid()) << clip.ts;
    ts_indexer.BuildIndex();
    mp4_indexer.BuildIndex();
    auto &index = ts_indexer.GetIndex();
    auto &expected = mp4_indexer.GetIndex();
    ASSERT_EQ(index.size(), expected.size()) << clip.ts;
    int keyframes = 0;
    for (size_t i = 0; i < index.size(); i++) {
      EXPECT_EQ(index[i].is_keyframe, expected[i].is_keyframe) << clip.ts << ": frame " << i;
      EXPECT_EQ(index[i].last_keyframe_id, expected[i].last_keyframe_id)
          << clip.ts << ": frame " << i;
      EXPECT_EQ(index[i].decode_idx, expected[i].decode_idx) << clip.ts << ": frame " << i;
      keyframes += index[i].is_keyframe;
    }
    EXPECT_EQ(keyframes, clip.keyframes) << clip.ts;
    CheckDecodeOrder(index, clip.ts);
  }
}

// A capture can start in the middle of a GOP. This MPEG-TS copy of cfr/test_1.mp4 starts at its
// packet 8, which is shown after the next one (frame 0), two packets before the first keyframe. A
// seek to frame 0, which is not a keyframe, is to land on the first packet: it takes the lowest
// decoding timestamp of the stream, not the one of frame 0. Only the index is checked: the pictures
// before the first keyframe refer to pictures that are not in the file.
TEST_F(FramesDecoderGpuTest, MpegTsIndexFromTheMiddleOfAGop) {
  auto path = make_string(testing::dali_extra_path(), "/db/video/mpegts/test_1_mid_gop.ts");
  FramesIndexer indexer(path);
  ASSERT_TRUE(indexer.IsValid()) << path;
  indexer.BuildIndex();
  auto &index = indexer.GetIndex();
  ASSERT_EQ(index.size(), 42u) << path;
  EXPECT_FALSE(index[0].is_keyframe);
  EXPECT_EQ(index[0].decode_idx, 1);
  EXPECT_EQ(index[1].decode_idx, 0);
  EXPECT_TRUE(index[2].is_keyframe);
  EXPECT_EQ(index.SeekDecodeIdx(0), 0);
  CheckDecodeOrder(index, path);
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
    bool indexer_valid = indexer.IsValid() && FramesDecoderGpu::IsCodecSupported(indexer.CodecId());
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

// Seeks in MPEG-TS (see MpegTsIndex), checked against the frames of the clips
TEST_F(FramesDecoderGpuTest, ConstantFrameRateMpegTs) {
  FramesDecoderGpu decoder(MpegTsClips()[0].ts);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRateHevcMpegTs) {
  if (!FramesDecoderGpu::SupportsHevc()) {
    GTEST_SKIP();
  }
  FramesDecoderGpu decoder(MpegTsClips()[1].ts);
  decoder.BuildIndex();
  RunTest(decoder, cfr_videos_[1]);
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
    bool indexer_valid = indexer.IsValid() && FramesDecoderCpu::IsCodecSupported(indexer.CodecId());
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

}  // namespace dali
