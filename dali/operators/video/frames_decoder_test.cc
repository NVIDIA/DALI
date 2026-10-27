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
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

// The frames of the Sintel clips: B-frames, some with hidden lead-ins, some with a single IDR
std::vector<std::string> SintelClips() {
  std::vector<std::string> paths;
  for (int clip = 0; clip < 10; clip++)
    paths.push_back(SintelClip(clip % 3, clip));
  for (int clip = 0; clip < 5; clip++)
    paths.push_back(SintelVideoFile(clip));
  return paths;
}

// One Sintel clip of each kind: no lead-in, only the IDR hidden (a single GOP), a seek that lands
// on the hidden IDR, seeks that land on the shown IDRs, and a long lead-in (188 hidden pictures).
// SintelClips() has more of the same kinds; the DISABLED_*AllSintelClips tests sweep them all.
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

// The GPU parser is sent packets without an end-of-picture flag, so it decodes a picture once the
// next packet arrives, but now and then (e.g. some B-frames of the Sintel clips) only with the one
// after: a decoder may have decoded one picture more than AddDecodeWork estimates, which its next
// frames then do not cost.
constexpr int kGpuPictureSlack = 1;

/**
 * @brief Decodes `frame_ids` as the reader does (padded by clamping) and checks that the decoder
 * decodes the pictures that AddDecodeWork estimates from `pos`, reopens the file as estimated,
 * and ends where `pos` does.
 *
 * The decoder may decode up to `slack` pictures more than estimated, which its next frames then do
 * not cost: `surplus` carries how many pictures it may be ahead from the previous samples.
 */
void CheckDecodeWork(FramesDecoderBase &decoder, FramesDecoderBase::DecodePosition &pos,
                     int64_t &surplus, const std::vector<int> &frame_ids, uint8_t *buffer,
                     int slack, const std::string &what) {
  ASSERT_LE(frame_ids.size(), static_cast<size_t>(kMaxDecodeWorkFrames));
  auto &index = decoder.GetIndex();
  int n = index.size();
  // the frames that DecodeFrames seeks to, in order
  std::vector<int> frames;
  for (int frame : frame_ids)
    frames.push_back(FramesDecoderBase::HandleBoundary(boundary::BoundaryType::CLAMP, frame, 0, n));
  std::sort(frames.begin(), frames.end());
  FramesDecoderBase::DecodeWork work;
  for (int frame : frames)
    FramesDecoderBase::AddDecodeWork(index, pos, frame, work);

  int64_t before = decoder.DecodedPictures();
  int64_t reopens_before = decoder.Reopens();
  decoder.DecodeFrames(buffer, make_cspan(frame_ids), boundary::BoundaryType::CLAMP);
  int64_t ahead = decoder.DecodedPictures() - before - work.pictures;
  EXPECT_GE(ahead, -surplus) << what;
  EXPECT_LE(ahead, slack) << what;
  // a seek drops the surplus, so this is an upper bound
  surplus = std::clamp<int64_t>(surplus + ahead, 0, slack);
  EXPECT_EQ(decoder.Reopens() - reopens_before, work.reopens) << what;
  EXPECT_EQ(decoder.NextFrameIdx(), pos.next_frame) << what;
}

/**
 * @brief Runs, on one decoder, samples that take every path of SeekFrame: decoding through,
 * seeking forward and backward, to the same and to another keyframe, and resetting.
 */
void CheckDecodeWorkScenarios(FramesDecoderBase &decoder, uint8_t *buffer, int slack,
                              const std::string &path) {
  auto &index = decoder.GetIndex();
  int n = index.size();
  ASSERT_GE(n, 40);
  FramesDecoderBase::DecodePosition pos;  // a new decoder
  int64_t surplus = 0;
  int a = std::min(kMaxDecodeWorkFrames, n / 5);
  int next = a;
  auto check = [&](const std::string &what, const std::vector<int> &frames) {
    CheckDecodeWork(decoder, pos, surplus, frames, buffer, slack, make_string(path, ": ", what));
    next = pos.next_frame;
  };
  check("fresh", FrameRange(0, a));
  check("continue", FrameRange(next, next + 8));
  // Early in the file, so that there is a keyframe ahead and room for the leaps after it
  int keyframe = next + 1;
  while (keyframe < n && !index[keyframe].is_keyframe)
    keyframe++;
  if (keyframe + 4 + 9 < n - 4)
    check("across a keyframe", FrameRange(keyframe + 1, keyframe + 4));
  static_assert(FramesDecoderBase::kMinimumSeekLeap >= 5);
  check("short leap", FrameRange(next + 5, next + 9));
  int far = std::min(next + 20, n - 4);  // not to the last frame: the next sample seeks backward
  check("far jump", FrameRange(far, far + 3));
  check("backward", FrameRange(5, 9));
  check("duplicate", {12, 12, 13});
  check("last frame, twice", {n - 1, n - 1});
  check("first frame after the end", {0, 1});
  std::vector<int> uniform;
  for (int i = 0; i < 8; i++)
    uniform.push_back(std::lround(i * (n - 1) / 7.0));
  check("uniform", uniform);
  check("edge-padded tail", FrameRange(n - 4, n + 3));
  check("from the start", FrameRange(0, 4));
}

/**
 * @brief Reads the frames in a row and checks that, from PositionAt, AddDecodeWork estimates the
 * pictures decoded for each frame: in total, the decoder is never behind the estimate and at most
 * `slack` pictures ahead. Returns the number of frames that cost nothing (decoded earlier, for a
 * frame shown before them).
 */
int CheckPositionAt(FramesDecoderBase &decoder, uint8_t *frame, int slack,
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
    EXPECT_LE(ahead, slack) << path << ": frame " << k;
    free_frames += work.pictures == 0;
  }
  return free_frames;
}

void CheckGPUDecodeWork(const std::vector<std::string> &paths) {
  for (auto &path : paths) {
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    DeviceBuffer<uint8_t> buffer;
    buffer.resize(kMaxDecodeWorkFrames * decoder.FrameSize());
    CheckDecodeWorkScenarios(decoder, buffer.data(), kGpuPictureSlack, path);
  }
}

void CheckGPUPositionAt(const std::vector<std::string> &paths) {
  for (auto &path : paths) {
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    DeviceBuffer<uint8_t> buffer;
    buffer.resize(decoder.FrameSize());
    EXPECT_GT(CheckPositionAt(decoder, buffer.data(), kGpuPictureSlack, path), 0)
        << path << ": no B-frames";
  }
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
  int n = index.size();
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

// Holds up the work enqueued on `stream` after this call for `ms` milliseconds
void DelayStream(cudaStream_t stream, int ms) {
  auto sleep = [](void *ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(reinterpret_cast<intptr_t>(ms)));
  };
  CUDA_CALL(cudaLaunchHostFunc(stream, sleep, reinterpret_cast<void *>(intptr_t{ms})));
}

constexpr int kConversionDelayMs = 5;

// Frames to read in this order: in a row, seeking forward and back, the last frame, from the start
// after the end and across a short leap (the skipped frames are decoded, but not returned)
std::vector<int> ReadOrder(int num_frames) {
  std::vector<int> frames = FrameRange(0, 10);
  for (int frame : {25, 26, 12, 13, num_frames - 1, 0, 1, 30, 33})
    frames.push_back(std::min(frame, num_frames - 1));
  return frames;
}

using GpuDecoderFactory = std::function<std::unique_ptr<FramesDecoderGpu>(cudaStream_t)>;

/**
 * @brief Reads `frame_ids` in this order (seeking to a frame that is not the next one) into `out`,
 * one frame after another, on the stream of the decoder.
 *
 * With `delay_ms`, the stream is held up before each frame is read, so that the conversions of the
 * pictures are still pending when the decoder decodes the next picture (or is destroyed). Checks
 * that at most `max_mapped` output surfaces are mapped in the process meanwhile.
 */
void ReadFrames(FramesDecoderGpu &decoder, cudaStream_t stream, const std::vector<int> &frame_ids,
                uint8_t *out, int delay_ms, int max_mapped = 1) {
  for (int frame : frame_ids) {
    if (decoder.NextFrameIdx() != frame)
      decoder.SeekFrame(frame);
    if (delay_ms > 0)
      DelayStream(stream, delay_ms);
    ASSERT_TRUE(decoder.ReadNextFrame(out)) << "frame " << frame;
    EXPECT_LE(frame_dec_gpu_impl::NVDECFramesMapped(), max_mapped);
    out += decoder.FrameSize();
  }
}

/**
 * @brief Decodes `frame_ids` (see ReadFrames) with a decoder on `stream` and returns them on the
 * host. The decoder is destroyed first, while the conversion of its last picture may be pending.
 */
std::vector<uint8_t> DecodeToHost(const GpuDecoderFactory &make_decoder, cudaStream_t stream,
                                  const std::vector<int> &frame_ids, int delay_ms) {
  auto decoder = make_decoder(stream);
  size_t frame_size = decoder->FrameSize();
  DeviceBuffer<uint8_t> frames;
  frames.resize(frame_ids.size() * frame_size);
  ReadFrames(*decoder, stream, frame_ids, frames.data(), delay_ms);
  decoder.reset();
  EXPECT_EQ(frame_dec_gpu_impl::NVDECFramesMapped(), 0) << "a surface was not unmapped";
  std::vector<uint8_t> host(frames.size());
  MemCopy(host.data(), frames.data(), host.size(), stream);
  CUDA_CALL(cudaStreamSynchronize(stream));
  return host;
}

/**
 * @brief Checks that the frames (see ReadOrder) decoded on a stream of their own, held up so that
 * the conversions are pending, are the same as those decoded on the default stream without the
 * delay, and match the ground truth.
 */
void CheckConversionPending(const GpuDecoderFactory &make_decoder, TestVideo &ground_truth,
                            double eps) {
  auto frame_ids = ReadOrder(ground_truth.NumFrames());
  auto reference = DecodeToHost(make_decoder, 0, frame_ids, 0);
  auto stream = CUDAStream::Create(true);
  auto frames = DecodeToHost(make_decoder, stream, frame_ids, kConversionDelayMs);
  ASSERT_EQ(frames.size(), reference.size());
  EXPECT_TRUE(frames == reference);
  size_t frame_size = ground_truth.FrameSize();
  for (size_t i = 0; i < frame_ids.size(); i++) {
    CompareFrameAvgError(frame_ids[i], frame_size, ground_truth.Width(), ground_truth.Height(),
                         frames.data() + i * frame_size, ground_truth.frames_[frame_ids[i]].data,
                         eps);
  }
}

GpuDecoderFactory FileDecoder(const std::string &path) {
  return [path](cudaStream_t stream) {
    auto decoder = std::make_unique<FramesDecoderGpu>(path, stream);
    decoder->BuildIndex();
    return decoder;
  };
}

}  // namespace

static_assert(FramesDecoderBase::kMinimumSeekLeap == 10, "the steps below assume it");

// Exact estimates on hand-built indices: B-frames, a hidden lead-in, a seek that lands before its
// keyframe and an open GOP, which the decoder tests only cover with real files and some slack
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
  for (auto &path : {cfr_videos_paths_[0], vfr_videos_paths_[0],
                     make_string(testing::dali_extra_path(), "/db/video/cfr/test_2_vp9.mp4"),
                     make_string(testing::dali_extra_path(), "/db/video/vfr/test_2_vp9.mp4")}) {
    FramesDecoderCpu decoder(path);
    decoder.BuildIndex();
    std::vector<uint8_t> buffer(kMaxDecodeWorkFrames * decoder.FrameSize());
    CheckDecodeWorkScenarios(decoder, buffer.data(), 0, path);
    decoder.Reset();
    CheckPositionAt(decoder, buffer.data(), 0, path);
  }
}

TEST_F(FramesDecoderGpuTest, DecodeWork) {
  std::vector<std::string> paths = SintelClipKinds();
  // VP9, H.264 without a lead-in and HEVC (the CPU test covers the other VP9 files)
  paths.push_back(vfr_videos_paths_[0]);
  paths.push_back(make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4"));
  if (FramesDecoderGpu::SupportsHevc())
    paths.push_back(vfr_hevc_videos_paths_[0]);
  CheckGPUDecodeWork(paths);
}

TEST_F(FramesDecoderGpuTest, DecodeWorkFromPosition) {
  CheckGPUPositionAt(SintelClipKinds());
}

// The same, on every Sintel clip: too slow to run by default (~30 s), so run it with
// --gtest_also_run_disabled_tests after changing SeekFrame or AddDecodeWork
TEST_F(FramesDecoderGpuTest, DISABLED_DecodeWorkAllSintelClips) {
  CheckGPUDecodeWork(SintelClips());
}

TEST_F(FramesDecoderGpuTest, DISABLED_DecodeWorkFromPositionAllSintelClips) {
  CheckGPUPositionAt(SintelClips());
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

TEST_F(FramesDecoderGpuTest, IndexDecodeOrder) {
  auto h264_path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4");
  for (auto &path : {cfr_videos_paths_[0], h264_path}) {
    FramesDecoderGpu decoder(path);
    decoder.BuildIndex();
    CheckDecodeOrder(decoder.GetIndex(), path);
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

// The GPU decoder does not wait for the conversion of a picture before it decodes the next one:
// held up on a stream of its own, the conversions are pending when the decoder decodes the next
// picture, seeks, flushes and is destroyed. The frames are the same as when nothing is pending.
TEST_F(FramesDecoderGpuTest, ConversionPendingAtNextPicture) {
  CheckConversionPending(FileDecoder(cfr_videos_paths_[0]), cfr_videos_[0], 1.5);
}

// The same without an index: the frames go through the frame buffer of the decoder
TEST_F(FramesDecoderGpuTest, ConversionPendingAtNextPictureNoIndex) {
  auto &ground_truth = vfr_videos_[1];
  auto memory_video = MemoryVideo(vfr_mpeg4_mkv_videos_paths_[1]);
  auto make_decoder = [&](cudaStream_t stream) {
    auto decoder =
        std::make_unique<FramesDecoderGpu>(memory_video.data(), memory_video.size(), "", stream);
    decoder->SetNumFrames(ground_truth.NumFrames());
    return decoder;
  };
  CheckConversionPending(make_decoder, ground_truth, 3.0);
}

// A decoder destroyed while the conversion of its last picture is pending unmaps the surface
// before NVDEC goes back to the cache, where the next decoder, on another stream, takes it
TEST_F(FramesDecoderGpuTest, DecoderReturnedToCacheIsClean) {
  std::vector<int> frame_ids = FrameRange(0, 8);
  frame_ids.push_back(20);
  std::vector<std::vector<uint8_t>> reference;
  for (int clip = 0; clip < 5; clip++)  // also puts a decoder for these clips in the cache
    reference.push_back(DecodeToHost(FileDecoder(SintelClip(clip % 3, clip)), 0, frame_ids, 0));
  size_t num_created = frame_dec_gpu_impl::NVDECDecodersCreated();
  CUDAStream streams[2] = {CUDAStream::Create(true), CUDAStream::Create(true)};
  for (int clip = 0; clip < 5; clip++) {
    auto frames = DecodeToHost(FileDecoder(SintelClip(clip % 3, clip)), streams[clip % 2],
                               frame_ids, kConversionDelayMs);
    EXPECT_TRUE(frames == reference[clip]) << "clip " << clip;
  }
  EXPECT_EQ(frame_dec_gpu_impl::NVDECDecodersCreated(), num_created);
}

// Decoders that run in threads of their own, on streams of their own, are destroyed in another
// thread (as the video reader does with the decoders of idle lanes), while the conversion of their
// last picture is likely pending
TEST_F(FramesDecoderGpuTest, DecodersDestroyedInAnotherThread) {
  constexpr int kThreads = 4;
  std::vector<int> frame_ids = FrameRange(0, 6);
  frame_ids.push_back(15);
  std::vector<std::vector<uint8_t>> reference;
  for (int clip = 0; clip < kThreads; clip++)
    reference.push_back(DecodeToHost(FileDecoder(SintelClip(clip % 3, clip)), 0, frame_ids, 0));

  std::vector<CUDAStream> streams;
  std::vector<std::unique_ptr<FramesDecoderGpu>> decoders(kThreads);
  std::vector<DeviceBuffer<uint8_t>> frames(kThreads);
  for (int t = 0; t < kThreads; t++)
    streams.push_back(CUDAStream::Create(true));
  auto run = [&](std::function<void(int)> body) {
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; t++) {
      threads.emplace_back([&, t]() {
        DeviceGuard guard(0);
        body(t);
      });
    }
    for (auto &thread : threads)
      thread.join();
  };
  run([&](int t) {
    decoders[t] = FileDecoder(SintelClip(t % 3, t))(streams[t]);
    frames[t].resize(frame_ids.size() * decoders[t]->FrameSize());
    ReadFrames(*decoders[t], streams[t], frame_ids, frames[t].data(), 20, kThreads);
  });
  run([&](int t) {
    decoders[(t + 1) % kThreads].reset();
  });
  EXPECT_EQ(frame_dec_gpu_impl::NVDECFramesMapped(), 0);
  for (int t = 0; t < kThreads; t++) {
    std::vector<uint8_t> host(frames[t].size());
    MemCopy(host.data(), frames[t].data(), host.size(), streams[t]);
    CUDA_CALL(cudaStreamSynchronize(streams[t]));
    EXPECT_TRUE(host == reference[t]) << "clip " << t;
  }
}

// A conversion that fails once the surface is mapped (here: for an output type that it does not
// support) does not leave the surface mapped when the decoder goes back to the cache
TEST_F(FramesDecoderGpuTest, SurfaceUnmappedAfterFailedConversion) {
  auto stream = CUDAStream::Create(true);
  {
    FramesDecoderGpu decoder(cfr_videos_paths_[0], stream);
    decoder.BuildIndex();
    decoder.SetOutputType(DALI_INT16);
    EXPECT_THROW(decoder.ReadNextFrame(FrameData()), std::exception);
    EXPECT_EQ(frame_dec_gpu_impl::NVDECFramesMapped(), 1);
  }
  EXPECT_EQ(frame_dec_gpu_impl::NVDECFramesMapped(), 0);
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
  std::string path = make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4");
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
