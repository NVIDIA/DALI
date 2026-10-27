// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/reader/decode_lane_planner.h"
#include "dali/test/dali_test_config.h"

namespace dali {

namespace {

constexpr int64_t kOpenCost = 16;

// An index of `num_frames` frames in decoding order, with a keyframe every `gop` frames
FrameIndex MakeIndex(int num_frames, int gop) {
  FrameIndex index;
  index.timebase = {1, 25};
  for (int i = 0; i < num_frames; i++)
    index.index.push_back({i, i / gop * gop, i % gop == 0, i == num_frames - 1});
  return index;
}

// A decoder of `file` that read the frames up to `next_frame - 1` in a row
LaneStart Held(std::string_view file, const FrameIndex &index, int next_frame) {
  return {file, FramesDecoderBase::PositionAt(index, next_frame)};
}

// Where a decoder with an index is, for LaneStart
FramesDecoderBase::DecodePosition DecoderPosition(FramesDecoderBase &decoder) {
  return FramesDecoderBase::PositionAt(decoder.GetIndex(), decoder.NextFrameIdx());
}

/**
 * @brief The runs of a batch, and the frames of their samples.
 */
struct Batch {
  std::vector<PlannedRun> runs;
  std::deque<std::vector<int>> frames;  // of the samples, referenced by the runs

  // Adds a run of samples, each the given frames
  void Add(std::string_view file, const FrameIndex *index, int batch_idx,
           const std::vector<std::vector<int>> &samples) {
    PlannedRun run{file, index, batch_idx, -1, -1, {}};
    for (auto &sample : samples) {
      frames.push_back(sample);
      std::sort(frames.back().begin(), frames.back().end());
      run.samples.push_back(make_cspan(frames.back()));
      for (int frame : sample) {
        run.first = run.first < 0 ? frame : std::min(run.first, frame);
        run.last = std::max(run.last, frame);
      }
    }
    runs.push_back(run);
  }

  // Adds a run of one sample, the frames from `first` to `last`
  void Add(std::string_view file, const FrameIndex *index, int batch_idx, int first, int last) {
    std::vector<int> sample;
    for (int frame = first; frame <= last; frame++)
      sample.push_back(frame);
    Add(file, index, batch_idx, {sample});
  }
};

std::vector<LanePlan> Plan(const Batch &batch, const std::vector<LaneStart> &lanes) {
  std::vector<LanePlan> plans;
  PlanLanes(make_cspan(batch.runs), make_cspan(lanes), kOpenCost, plans);
  EXPECT_EQ(plans.size(), lanes.size());
  return plans;
}

/**
 * @brief The order in which the reader decoded a lane's runs before the planner was moved out of
 * it: the files by their first sample in the batch, the decoder's file first, the runs of a file
 * by first frame, and the run that continues where the decoder is moved to the front.
 */
std::vector<int> ReferenceOrder(const std::vector<PlannedRun> &runs_, const LaneStart &start,
                                std::vector<int> runs) {
  auto continues = [&](int r) {
    const auto &run = runs_[r];
    return run.filename == start.filename &&
           CanReachWithoutSeek(run.index, start.pos.next_frame, run.first);
  };
  std::map<std::string_view, int> file_order;
  for (int r : runs) {
    auto [it, inserted] = file_order.emplace(runs_[r].filename, runs_[r].batch_idx);
    if (!inserted)
      it->second = std::min(it->second, runs_[r].batch_idx);
  }
  auto held = file_order.find(start.filename);
  if (held != file_order.end())
    held->second = -1;
  std::sort(runs.begin(), runs.end(), [&](int a, int b) {
    const auto &ra = runs_[a], &rb = runs_[b];
    return std::make_tuple(file_order[ra.filename], ra.filename, ra.first) <
           std::make_tuple(file_order[rb.filename], rb.filename, rb.first);
  });
  auto it = std::find_if(runs.begin(), runs.end(), continues);
  if (it != runs.end())
    std::rotate(runs.begin(), it, it + 1);
  return runs;
}

std::vector<int> Frames(const std::vector<int> &frame_idxs, int start, int end, int stride,
                        boundary::BoundaryType boundary_type, int roi_start, int roi_end) {
  SmallVector<int, 32> frames = {-100};  // cleared
  SampleFrames(make_cspan(frame_idxs), start, end, stride, boundary_type, roi_start, roi_end,
               frames);
  return std::vector<int>(frames.begin(), frames.end());
}

/**
 * @brief A sample of the reader: the frames from `start` to `end` by `stride`, or `frame_idxs`,
 * padded by clamping to the file.
 */
struct ReaderSample {
  std::string path;
  int start, end, stride;
  std::vector<int> frame_idxs = {};
};

constexpr int kMaxSampleFrames = 16;

/**
 * @brief A lane of the reader, with the same decoder from batch to batch.
 */
struct ReaderLane {
  std::unique_ptr<FramesDecoderCpu> decoder;
  std::string filename;  // of the decoder
};

/**
 * @brief Plans a batch as the reader does, one run per sample, then decodes it, lane by lane, as
 * the reader does, and checks that the lanes decode the pictures planned.
 */
void CheckPlannedPictures(const std::vector<ReaderSample> &batch, std::vector<ReaderLane> &lanes,
                          const std::map<std::string, FrameIndex> &indices,
                          std::vector<uint8_t> &buffer, const std::string &what) {
  std::deque<SmallVector<int, 32>> frames;
  std::vector<PlannedRun> runs;
  for (int i = 0; i < static_cast<int>(batch.size()); i++) {
    auto &sample = batch[i];
    auto &index = indices.at(sample.path);
    frames.emplace_back();
    SampleFrames(make_cspan(sample.frame_idxs), sample.start, sample.end, sample.stride,
                 boundary::BoundaryType::CLAMP, 0, index.size(), frames.back());
    ASSERT_FALSE(frames.back().empty());
    runs.push_back({sample.path, &index, i, frames.back().front(), frames.back().back(),
                    {make_cspan(frames.back())}});
  }
  std::vector<LaneStart> starts(lanes.size());
  for (size_t l = 0; l < lanes.size(); l++) {
    if (lanes[l].decoder)
      starts[l] = {lanes[l].filename, DecoderPosition(*lanes[l].decoder)};
  }
  std::vector<LanePlan> plans;
  PlanLanes(make_cspan(runs), make_cspan(starts), kOpenCost, plans);

  size_t planned_runs = 0;
  for (size_t l = 0; l < lanes.size(); l++) {
    auto &lane = lanes[l];
    int64_t decoded = 0;
    for (int r : plans[l].runs) {
      auto &sample = batch[r];
      if (!lane.decoder || lane.filename != sample.path) {
        lane.decoder = std::make_unique<FramesDecoderCpu>(sample.path);
        lane.decoder->SetIndex(indices.at(sample.path));
        lane.filename = sample.path;
      }
      auto &decoder = *lane.decoder;
      int64_t before = decoder.DecodedPictures();
      if (sample.frame_idxs.empty()) {
        ASSERT_LE((sample.end - sample.start + sample.stride - 1) / sample.stride,
                  kMaxSampleFrames);
        decoder.DecodeFrames(buffer.data(), sample.start, sample.end, sample.stride,
                             boundary::BoundaryType::CLAMP);
      } else {
        ASSERT_LE(sample.frame_idxs.size(), static_cast<size_t>(kMaxSampleFrames));
        decoder.DecodeFrames(buffer.data(), make_cspan(sample.frame_idxs),
                             boundary::BoundaryType::CLAMP);
      }
      decoded += decoder.DecodedPictures() - before;
    }
    EXPECT_EQ(decoded, plans[l].pictures) << what << ", lane " << l;
    EXPECT_GE(plans[l].load - plans[l].pictures, 0) << what << ", lane " << l;
    EXPECT_EQ((plans[l].load - plans[l].pictures) % kOpenCost, 0) << what << ", lane " << l;
    planned_runs += plans[l].runs.size();
  }
  EXPECT_EQ(planned_runs, runs.size()) << what;
}

}  // namespace

TEST(DecodeLanePlannerTest, SampleFrames) {
  using boundary::BoundaryType;
  // the given frames (uniform sampling), sorted, duplicates kept
  EXPECT_EQ(Frames({7, 3, 9, 3}, 0, 0, 1, BoundaryType::CLAMP, 0, 10),
            std::vector<int>({3, 3, 7, 9}));
  // a range, with a stride
  EXPECT_EQ(Frames({}, 2, 10, 3, BoundaryType::ISOLATED, 0, 10), std::vector<int>({2, 5, 8}));
  // padded past the end of the file
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::CLAMP, 0, 10),
            std::vector<int>({7, 8, 9, 9, 9, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::REFLECT_101, 0, 10),
            std::vector<int>({6, 7, 7, 8, 8, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::REFLECT_1001, 0, 10),
            std::vector<int>({7, 7, 8, 8, 9, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::CONSTANT, 0, 10), std::vector<int>({7, 8, 9}));
  EXPECT_EQ(Frames({}, 10, 13, 1, BoundaryType::CONSTANT, 0, 10), std::vector<int>());
  // a region of the file: clamped to it
  EXPECT_EQ(Frames({}, 18, 22, 1, BoundaryType::CLAMP, 20, 30),
            std::vector<int>({20, 20, 20, 21}));
  EXPECT_EQ(Frames({}, 28, 32, 1, BoundaryType::CLAMP, 20, 30),
            std::vector<int>({28, 29, 29, 29}));
  EXPECT_THROW(Frames({}, 7, 13, 1, BoundaryType::ISOLATED, 0, 10), std::exception);
}

TEST(DecodeLanePlannerTest, CanReachWithoutSeek) {
  auto index = MakeIndex(100, 30);
  EXPECT_TRUE(CanReachWithoutSeek(&index, 5, 5));
  EXPECT_TRUE(CanReachWithoutSeek(&index, 5, 5 + FramesDecoderBase::kMinimumSeekLeap));
  EXPECT_TRUE(CanReachWithoutSeek(&index, 5, 29));   // the same keyframe
  EXPECT_FALSE(CanReachWithoutSeek(&index, 5, 31));  // the next keyframe
  EXPECT_FALSE(CanReachWithoutSeek(&index, 5, 4));   // backward
  EXPECT_FALSE(CanReachWithoutSeek(&index, -1, 0));  // after the end
  EXPECT_FALSE(CanReachWithoutSeek(nullptr, 5, 29));
}

TEST(DecodeLanePlannerTest, LongestFirstToEarliestFinish) {
  auto index = MakeIndex(100, 100);
  // three files, a run of each from frame 0: 30, 20 and 10 frames
  Batch batch;
  batch.Add("b", &index, 1, 0, 19);
  batch.Add("a", &index, 0, 0, 29);
  batch.Add("c", &index, 2, 0, 9);
  auto plans = Plan(batch, {{}, {}});
  // a: lane 0 (a tie, to the lowest lane), b: lane 1, c: lane 1 (36 + 26 < 46 + 26)
  EXPECT_EQ(plans[0].runs, std::vector<int>({1}));
  EXPECT_EQ(plans[0].load, 30 + kOpenCost);
  EXPECT_EQ(plans[1].runs, std::vector<int>({0, 2}));
  EXPECT_EQ(plans[1].load, 20 + 10 + 2 * kOpenCost);
}

TEST(DecodeLanePlannerTest, HeldFileContinues) {
  auto index = MakeIndex(100, 50);
  // frames 20-39 of the GOP from frame 0: 20 frames of preroll after a seek
  Batch batch;
  batch.Add("a", &index, 0, 20, 39);
  auto plans = Plan(batch, {{"b", {0, -1}}, Held("a", index, 20)});
  EXPECT_TRUE(plans[0].runs.empty());
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, 20);  // no opening, no preroll
  // a decoder that has to seek back pays the preroll, but not the opening
  plans = Plan(batch, {{"b", {0, -1}}, Held("a", index, 45)});
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, 20 + 20);
}

TEST(DecodeLanePlannerTest, NoIndex) {
  // without an index, a run costs its frames (and the opening), with no preroll
  Batch batch;
  batch.Add("a", nullptr, 0, 20, 39);
  auto plans = Plan(batch, {{}});
  EXPECT_EQ(plans[0].load, 20 + kOpenCost);
  plans = Plan(batch, {{"a", {45, -1}}});
  EXPECT_EQ(plans[0].load, 20);
}

TEST(DecodeLanePlannerTest, HeldFileGoesFirst) {
  // A lane that holds X gets a run of Y, the longer one, and then the run of X that continues
  // where its decoder is. It decodes X first, on its decoder, and then opens Y.
  auto index = MakeIndex(100, 100);
  Batch batch;
  batch.Add("y", &index, 0, 0, 29);
  batch.Add("x", &index, 1, 10, 19);
  auto plans = Plan(batch, {Held("x", index, 10)});
  EXPECT_EQ(plans[0].runs, std::vector<int>({1, 0}));
  EXPECT_EQ(plans[0].load, 10 + kOpenCost + 30);  // not reopening X after Y
}

TEST(DecodeLanePlannerTest, RunsOfAFileInAscendingOrder) {
  // Two runs of a file with a single GOP: the lane decodes them in ascending order, through from
  // the start, not the later one first and then back from the start again
  auto index = MakeIndex(200, 200);
  Batch batch;
  batch.Add("a", &index, 0, 100, 109);
  batch.Add("a", &index, 1, 0, 9);
  auto plans = Plan(batch, {{}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({1, 0}));
  EXPECT_EQ(plans[0].load, kOpenCost + 110);
}

TEST(DecodeLanePlannerTest, LeadIn) {
  // 20 packets hidden by an edit list before the first frame, which is not a keyframe: every
  // frame is decoded from the start of the stream
  FrameIndex index;
  index.timebase = {1, 25};
  for (int i = 0; i < 50; i++)
    index.index.push_back({i, 0, false, i == 49, i + 20, i == 0 ? 0 : -1});
  Batch batch;
  batch.Add("a", &index, 0, 30, 39);
  auto plans = Plan(batch, {{}});
  EXPECT_EQ(plans[0].load, kOpenCost + 60);
  // seeking back from frame 45 also starts from the hidden packets
  plans = Plan(batch, {Held("a", index, 45)});
  EXPECT_EQ(plans[0].load, 60);
}

TEST(DecodeLanePlannerTest, ResetAfterTheLastFrame) {
  // Edge padding repeats the last frame; a decoder that returned it reopens the file to return
  // it again, and decodes from the start
  auto index = MakeIndex(50, 50);
  Batch batch;
  batch.Add("a", &index, 0, {{45, 46, 47, 48, 49, 49, 49}});
  auto plans = Plan(batch, {{}});
  EXPECT_EQ(plans[0].load, kOpenCost + 50 + 2 * (kOpenCost + 50));
}

TEST(DecodeLanePlannerTest, LaneOrder) {
  // Random batches: runs of a few files with distinct first frames (no ties in the order)
  std::mt19937 rng(1234);
  auto index = MakeIndex(300, 30);
  std::vector<std::string> files = {"a", "b", "c", "d"};
  for (int iter = 0; iter < 200; iter++) {
    Batch batch;
    int num_runs = std::uniform_int_distribution<>(1, 12)(rng);
    std::vector<int> batch_idx(num_runs);
    for (int i = 0; i < num_runs; i++)
      batch_idx[i] = i;
    std::shuffle(batch_idx.begin(), batch_idx.end(), rng);
    std::map<std::string_view, std::vector<int>> firsts;
    for (int r = 0; r < num_runs; r++) {
      std::string_view file = files[rng() % files.size()];
      int first;
      do {
        first = rng() % 280;
      } while (std::count(firsts[file].begin(), firsts[file].end(), first));
      firsts[file].push_back(first);
      batch.Add(file, &index, batch_idx[r], first, first + static_cast<int>(rng() % 20));
    }
    std::vector<LaneStart> lanes(std::uniform_int_distribution<>(1, 4)(rng));
    for (auto &lane : lanes) {
      if (rng() % 3)
        lane = Held(files[rng() % files.size()], index, rng() % 300);
    }
    auto plans = Plan(batch, lanes);
    std::vector<int> planned;
    for (size_t l = 0; l < lanes.size(); l++) {
      EXPECT_EQ(plans[l].runs, ReferenceOrder(batch.runs, lanes[l], plans[l].runs))
          << "iteration " << iter << ", lane " << l;
      planned.insert(planned.end(), plans[l].runs.begin(), plans[l].runs.end());
    }
    std::sort(planned.begin(), planned.end());
    ASSERT_EQ(planned.size(), batch.runs.size());
    for (int r = 0; r < num_runs; r++)
      EXPECT_EQ(planned[r], r);
  }
}

TEST(DecodeLanePlannerTest, PlannedPicturesAreDecoded) {
  // VP9 (the codecs of the CPU decoder): 60 frames with a keyframe every 10, and 50 frames with
  // a single keyframe
  std::string gops = testing::dali_extra_path() + "/db/video/vfr/test_2_vp9.mp4";
  std::string single_gop = testing::dali_extra_path() + "/db/video/cfr/test_1_vp9.mp4";
  std::map<std::string, FrameIndex> indices;
  size_t frame_size = 0;
  for (auto &path : {gops, single_gop}) {
    FramesDecoderCpu decoder(path);
    decoder.BuildIndex();
    indices[path] = decoder.GetIndex();
    frame_size = std::max<size_t>(frame_size, decoder.FrameSize());
    ASSERT_GE(decoder.NumFrames(), 50) << path;
  }
  std::vector<uint8_t> buffer(kMaxSampleFrames * frame_size);
  int n = indices[gops].size();

  for (int num_lanes : {1, 2}) {
    // the first lane holds a decoder of the first file, which decoded frames 20 to 29
    std::vector<ReaderLane> lanes(num_lanes);
    lanes[0].decoder = std::make_unique<FramesDecoderCpu>(gops);
    lanes[0].decoder->SetIndex(indices[gops]);
    lanes[0].filename = gops;
    lanes[0].decoder->DecodeFrames(buffer.data(), 20, 30, 1);

    // continuing on that decoder, the last frame repeated (edge padding: the decoder resets), a
    // stride, uniform sampling, seeking forward and back, and continuing from the previous batch
    std::string what = make_string(num_lanes, " lanes, batch ");
    CheckPlannedPictures({{gops, n - 6, n + 2, 1},
                          {gops, 30, 46, 2},
                          {gops, 0, 0, 1, {0, n / 3, 2 * n / 3, n - 1}},
                          {single_gop, 0, 16, 1},
                          {single_gop, 25, 41, 1}},
                         lanes, indices, buffer, what + "1");
    CheckPlannedPictures({{gops, 30, 46, 1},
                          {gops, 5, 21, 1},
                          {single_gop, 16, 32, 1},
                          {single_gop, 34, 50, 1}},
                         lanes, indices, buffer, what + "2");
  }
}

}  // namespace dali
