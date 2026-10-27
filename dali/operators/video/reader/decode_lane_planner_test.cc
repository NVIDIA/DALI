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
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/util.h"
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
FramesDecoderBase::DecodePosition DecoderPosition(const FramesDecoderBase &decoder) {
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
    PlannedRun run{file, index, batch_idx, {}};
    for (auto &sample : samples) {
      frames.push_back(sample);
      std::sort(frames.back().begin(), frames.back().end());
      run.samples.push_back(make_cspan(frames.back()));
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
  PlanLanes(plans, make_cspan(batch.runs), make_cspan(lanes), kOpenCost);
  EXPECT_EQ(plans.size(), lanes.size());
  return plans;
}

/**
 * @brief An independent statement of the order PlanLanes documents for a lane's runs: the files by
 * their first sample in the batch, the decoder's file first, the runs of a file by first frame,
 * and the run that continues where the decoder is moved to the front.
 */
std::vector<int> ReferenceOrder(const std::vector<PlannedRun> &runs_, const LaneStart &start,
                                std::vector<int> runs) {
  // the smallest frame of a run; -1 if none
  auto first = [&](int r) {
    int frame = -1;
    for (auto sample : runs_[r].samples) {
      for (int f : sample)
        frame = frame < 0 ? f : std::min(frame, f);
    }
    return frame;
  };
  auto continues = [&](int r) {
    const auto &run = runs_[r];
    return !start.filename.empty() && run.filename == start.filename &&
           CanReachWithoutSeek(*run.index, start.pos.next_frame, first(r));
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
    return std::make_tuple(file_order[ra.filename], ra.filename, first(a)) <
           std::make_tuple(file_order[rb.filename], rb.filename, first(b));
  });
  auto it = std::find_if(runs.begin(), runs.end(), continues);
  if (it != runs.end())
    std::rotate(runs.begin(), it, it + 1);
  return runs;
}

std::vector<int> Frames(const std::vector<int> &frame_idxs, int start, int end, int stride,
                        boundary::BoundaryType boundary_type, int roi_start, int roi_end,
                        int num_frames) {
  SmallVector<int, 32> frames = {-100};  // cleared
  SampleFrames(frames, make_cspan(frame_idxs), start, end, stride, boundary_type, roi_start,
               roi_end, num_frames);
  return std::vector<int>(frames.begin(), frames.end());
}

std::vector<int> Region(const std::vector<int> &frame_idxs, int start, int end, int stride,
                        boundary::BoundaryType boundary_type, int roi_start, int roi_end) {
  SmallVector<int, 32> frames = {-100};  // cleared
  RegionFrames(frames, make_cspan(frame_idxs), start, end, stride, boundary_type, roi_start,
               roi_end);
  return std::vector<int>(frames.begin(), frames.end());
}

/**
 * @brief A sample of the reader: the frames from `start` to `end` by `stride`, or `frame_idxs`,
 * padded to the region [roi_start, roi_end) of the file (by default, the whole file).
 */
struct ReaderSample {
  std::string path;
  int start, end, stride;
  std::vector<int> frame_idxs = {};
  boundary::BoundaryType boundary_type = boundary::BoundaryType::CLAMP;
  int roi_start = 0;
  int roi_end = -1;  // -1: the end of the file
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
 * @brief Plans a batch, one run per sample, then decodes it, lane by lane, as the reader does (the
 * frames of RegionFrames, through DecodeFrames), and checks that the lanes decode the pictures
 * planned. (The reader can also join samples into one run, when a decoder reaches one from the
 * other without seeking.)
 */
void CheckPlannedPictures(const std::vector<ReaderSample> &batch, std::vector<ReaderLane> &lanes,
                          const std::map<std::string, FrameIndex> &indices,
                          std::vector<uint8_t> &buffer, const std::string &what) {
  std::deque<SmallVector<int, 32>> frames;
  std::vector<PlannedRun> runs;
  for (int i = 0; i < static_cast<int>(batch.size()); i++) {
    auto &sample = batch[i];
    auto &index = indices.at(sample.path);
    int roi_end = sample.roi_end < 0 ? index.size() : sample.roi_end;
    frames.emplace_back();
    SampleFrames(frames.back(), make_cspan(sample.frame_idxs), sample.start, sample.end,
                 sample.stride, sample.boundary_type, sample.roi_start, roi_end, index.size());
    ASSERT_FALSE(frames.back().empty());
    runs.push_back({sample.path, &index, i, {make_cspan(frames.back())}});
  }
  std::vector<LaneStart> starts(lanes.size());
  for (size_t l = 0; l < lanes.size(); l++) {
    if (lanes[l].decoder)
      starts[l] = {lanes[l].filename, DecoderPosition(*lanes[l].decoder)};
  }
  std::vector<LanePlan> plans;
  PlanLanes(plans, make_cspan(runs), make_cspan(starts), kOpenCost);

  size_t planned_runs = 0;
  SmallVector<int, 32> region;
  for (size_t l = 0; l < lanes.size(); l++) {
    auto &lane = lanes[l];
    // the pictures decoded, and the openings: new decoders, and the reopens of DecodeFrames (e.g.
    // to decode the last frame again)
    int64_t decoded = 0, openings = 0;
    for (int r : plans[l].runs) {
      auto &sample = batch[r];
      auto &index = indices.at(sample.path);
      if (!lane.decoder || lane.filename != sample.path) {
        lane.decoder = std::make_unique<FramesDecoderCpu>(sample.path);
        lane.decoder->SetIndex(index);
        lane.filename = sample.path;
        openings++;
      }
      auto &decoder = *lane.decoder;
      int64_t before = decoder.DecodedPictures();
      int64_t reopens_before = decoder.Reopens();
      int roi_end = sample.roi_end < 0 ? index.size() : sample.roi_end;
      RegionFrames(region, make_cspan(sample.frame_idxs), sample.start, sample.end, sample.stride,
                   sample.boundary_type, sample.roi_start, roi_end);
      ASSERT_LE(region.size(), static_cast<size_t>(kMaxSampleFrames));
      std::vector<uint8_t> constant_frame(decoder.FrameSize());
      decoder.DecodeFrames(buffer.data(), make_cspan(region), sample.boundary_type,
                           constant_frame.data());
      decoded += decoder.DecodedPictures() - before;
      openings += decoder.Reopens() - reopens_before;
    }
    EXPECT_EQ(decoded, plans[l].pictures) << what << ", lane " << l;
    EXPECT_EQ(plans[l].load - plans[l].pictures, openings * kOpenCost) << what << ", lane " << l;
    planned_runs += plans[l].runs.size();
  }
  EXPECT_EQ(planned_runs, runs.size()) << what;
}

}  // namespace

TEST(DecodeLanePlannerTest, EvenSplitLanes) {
  // 16 samples: 3 rounds on 7 lanes as on 6; 2 rounds on 14 lanes as on 8; 1 round on 16
  EXPECT_EQ(EvenSplitLanes(7, 16), 6);
  EXPECT_EQ(EvenSplitLanes(14, 16), 8);
  EXPECT_EQ(EvenSplitLanes(16, 16), 16);
  EXPECT_EQ(EvenSplitLanes(21, 16), 16);  // capped at the batch size
  EXPECT_EQ(EvenSplitLanes(8, 16), 8);
  EXPECT_EQ(EvenSplitLanes(4, 8), 4);
  EXPECT_EQ(EvenSplitLanes(3, 8), 3);  // 3 rounds on 3 lanes; 2 lanes would need 4
  EXPECT_EQ(EvenSplitLanes(5, 8), 4);  // 2 rounds on 5 lanes as on 4
  EXPECT_EQ(EvenSplitLanes(12, 32), 11);  // 3 rounds
  // never fewer than one lane, never more than the batch
  EXPECT_EQ(EvenSplitLanes(0, 16), 1);
  EXPECT_EQ(EvenSplitLanes(8, 1), 1);
  EXPECT_EQ(EvenSplitLanes(8, 0), 1);
  for (int batch = 1; batch <= 64; batch++) {
    for (int max_lanes = 1; max_lanes <= 64; max_lanes++) {
      int lanes = EvenSplitLanes(max_lanes, batch);
      int cap = std::min(max_lanes, batch);
      ASSERT_GE(lanes, 1);
      ASSERT_LE(lanes, cap);
      // as many rounds as with `cap` lanes, and one lane fewer would need more
      auto rounds = [&](int n) { return div_ceil(batch, n); };
      ASSERT_EQ(rounds(lanes), rounds(cap)) << batch << " " << max_lanes;
      if (lanes > 1) {
        ASSERT_GT(rounds(lanes - 1), rounds(cap)) << batch << " " << max_lanes;
      }
    }
  }
}

TEST(DecodeLanePlannerTest, SampleFrames) {
  using boundary::BoundaryType;
  // the given frames (uniform sampling), sorted, duplicates kept
  EXPECT_EQ(Frames({7, 3, 9, 3}, 0, 0, 1, BoundaryType::CLAMP, 0, 10, 10),
            std::vector<int>({3, 3, 7, 9}));
  // a range, with a stride
  EXPECT_EQ(Frames({}, 2, 10, 3, BoundaryType::ISOLATED, 0, 10, 10), std::vector<int>({2, 5, 8}));
  // padded past the end of the file
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::CLAMP, 0, 10, 10),
            std::vector<int>({7, 8, 9, 9, 9, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::REFLECT_101, 0, 10, 10),
            std::vector<int>({6, 7, 7, 8, 8, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::REFLECT_1001, 0, 10, 10),
            std::vector<int>({7, 7, 8, 8, 9, 9}));
  EXPECT_EQ(Frames({}, 7, 13, 1, BoundaryType::CONSTANT, 0, 10, 10), std::vector<int>({7, 8, 9}));
  EXPECT_EQ(Frames({}, 10, 13, 1, BoundaryType::CONSTANT, 0, 10, 10), std::vector<int>());
  // a region of the file: clamped to it
  EXPECT_EQ(Frames({}, 18, 22, 1, BoundaryType::CLAMP, 20, 30, 100),
            std::vector<int>({20, 20, 20, 21}));
  EXPECT_EQ(Frames({}, 28, 32, 1, BoundaryType::CLAMP, 20, 30, 100),
            std::vector<int>({28, 29, 29, 29}));
  EXPECT_THROW(Frames({}, 7, 13, 1, BoundaryType::ISOLATED, 0, 10, 10), std::exception);
  // a stride that is not positive
  EXPECT_THROW(Frames({}, 0, 10, 0, BoundaryType::CLAMP, 0, 10, 10), std::invalid_argument);
}

TEST(DecodeLanePlannerTest, RegionFrames) {
  using boundary::BoundaryType;
  // the given frames, in their order, duplicates kept
  EXPECT_EQ(Region({7, 3, 9, 3}, 0, 0, 1, BoundaryType::CLAMP, 0, 10),
            std::vector<int>({7, 3, 9, 3}));
  // a range, with a stride, padded to the region, in the sample's order; -1 for a constant frame
  EXPECT_EQ(Region({}, 18, 30, 3, BoundaryType::CLAMP, 20, 26), std::vector<int>({20, 21, 24, 25}));
  EXPECT_EQ(Region({}, 18, 30, 3, BoundaryType::CONSTANT, 20, 26),
            std::vector<int>({-1, 21, 24, -1}));
  EXPECT_EQ(Region({}, 20, 30, 3, BoundaryType::REFLECT_101, 20, 26),
            std::vector<int>({20, 23, 24, 21}));
  // what SampleFrames sorts, after padding to the file
  EXPECT_EQ(Frames({}, 18, 30, 3, BoundaryType::CONSTANT, 20, 26, 100), std::vector<int>({21, 24}));
  EXPECT_THROW(Region({}, 18, 30, 3, BoundaryType::ISOLATED, 20, 26), std::exception);
  EXPECT_THROW(Region({}, 0, 10, -1, BoundaryType::CLAMP, 0, 10), std::invalid_argument);
}

TEST(DecodeLanePlannerTest, SampleFramesRegionOutsideOfTheFile) {
  using boundary::BoundaryType;
  // SampleFrames does not need the region to be within the file. A region [5, 14) of a file of 10
  // frames: frames 10 to 13 are in the region, and DecodeFrames pads them to the file
  EXPECT_EQ(Frames({}, 8, 14, 1, BoundaryType::CLAMP, 5, 14, 10),
            std::vector<int>({8, 9, 9, 9, 9, 9}));
  EXPECT_EQ(Frames({}, 8, 14, 1, BoundaryType::REFLECT_101, 5, 14, 10),
            std::vector<int>({5, 6, 7, 8, 8, 9}));
  EXPECT_EQ(Frames({}, 8, 14, 1, BoundaryType::CONSTANT, 5, 14, 10), std::vector<int>({8, 9}));
  EXPECT_EQ(Frames({12, 6}, 0, 0, 1, BoundaryType::CLAMP, 5, 14, 10), std::vector<int>({6, 9}));
  EXPECT_THROW(Frames({}, 8, 14, 1, BoundaryType::ISOLATED, 5, 14, 10), std::exception);
  // a region that starts before the file
  EXPECT_EQ(Frames({}, -3, 2, 1, BoundaryType::CLAMP, -3, 10, 10),
            std::vector<int>({0, 0, 0, 0, 1}));
}

TEST(DecodeLanePlannerTest, CanReachWithoutSeek) {
  auto index = MakeIndex(100, 30);
  EXPECT_TRUE(CanReachWithoutSeek(index, 5, 5));
  EXPECT_TRUE(CanReachWithoutSeek(index, 5, 5 + FramesDecoderBase::kMinimumSeekLeap));
  EXPECT_TRUE(CanReachWithoutSeek(index, 5, 29));   // the same keyframe
  EXPECT_FALSE(CanReachWithoutSeek(index, 5, 31));  // the next keyframe
  EXPECT_FALSE(CanReachWithoutSeek(index, 5, 4));   // backward
  EXPECT_FALSE(CanReachWithoutSeek(index, -1, 0));  // after the end
  EXPECT_TRUE(CanReachWithoutSeek(index, 25, 31));  // past a keyframe, but a short leap
  // a short leap, but outside of the index
  EXPECT_FALSE(CanReachWithoutSeek(index, 95, 103));
  EXPECT_FALSE(CanReachWithoutSeek(index, 100, 105));
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

// A batch of many runs over several files: every run is planned once, and the most loaded lane
// is within the bound of longest-first, earliest-finish assignment: the average load of the lanes
// plus the work of the longest run.
TEST(DecodeLanePlannerTest, BalancesALargeBatch) {
  constexpr int kNumFiles = 12, kNumRuns = 64;
  std::vector<FrameIndex> indices;
  std::vector<std::string> files;
  for (int f = 0; f < kNumFiles; f++) {
    indices.push_back(MakeIndex(1000, 50));
    files.push_back(make_string("file", f));
  }
  std::mt19937 rng(1234);
  Batch batch;
  for (int r = 0; r < kNumRuns; r++) {
    int f = rng() % kNumFiles;
    int length = 8 + rng() % 57;  // 8 to 64 frames
    int first = rng() % (1000 - length);
    batch.Add(files[f], &indices[f], r, first, first + length - 1);
  }
  // the work of each run alone, on a newly opened decoder
  int64_t longest = 0;
  for (int r = 0; r < kNumRuns; r++) {
    Batch one;
    one.runs.push_back(batch.runs[r]);
    longest = std::max(longest, Plan(one, {{}})[0].load);
  }
  for (int num_lanes : {4, 8}) {
    SCOPED_TRACE(make_string(num_lanes, " lanes"));
    auto plans = Plan(batch, std::vector<LaneStart>(num_lanes));
    std::vector<int> planned;
    int64_t total = 0, max_load = 0;
    for (auto &plan : plans) {
      planned.insert(planned.end(), plan.runs.begin(), plan.runs.end());
      total += plan.load;
      max_load = std::max(max_load, plan.load);
      EXPECT_GT(plan.load, 0) << "a lane without work, with more runs than lanes";
    }
    std::sort(planned.begin(), planned.end());
    std::vector<int> all(kNumRuns);
    std::iota(all.begin(), all.end(), 0);
    EXPECT_EQ(planned, all);
    EXPECT_LE(max_load, div_ceil(total, num_lanes) + longest);
    // The bound above is loose: assigning the runs in batch order instead of longest first also
    // meets it (1069 and 585). So the plan of this batch is pinned too
    EXPECT_EQ(max_load, num_lanes == 4 ? 1046 : 535);
  }
}

// A run is never split between lanes: consecutive samples of one long video, read in order, are
// one run, which one lane decodes while the other lanes have nothing to do. Here 32 samples of 16
// frames with a keyframe every 250 frames: split at the keyframe 250, two lanes would decode 250
// and 262 pictures, but one lane decodes all 512.
TEST(DecodeLanePlannerTest, OneLongRunUsesOneLane) {
  auto index = MakeIndex(1000, 250);
  std::vector<std::vector<int>> samples;
  for (int s = 0; s < 32; s++) {
    samples.emplace_back();
    for (int frame = 16 * s; frame < 16 * (s + 1); frame++)
      samples.back().push_back(frame);
  }
  Batch batch;
  batch.Add("a", &index, 0, samples);
  auto plans = Plan(batch, std::vector<LaneStart>(4));
  EXPECT_EQ(plans[0].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[0].load, 512 + kOpenCost);
  for (int l = 1; l < 4; l++)
    EXPECT_TRUE(plans[l].runs.empty()) << "lane " << l;
}

TEST(DecodeLanePlannerTest, TiesToTheLeastBacklog) {
  auto index = MakeIndex(100, 100);
  Batch batch;
  batch.Add("a", &index, 0, 0, 29);
  // lanes without decoders, which would all finish the run at the same time: the lowest one
  auto plans = Plan(batch, {{}, {}, {}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({0}));
  // unless it has more work of earlier batches left than another
  LaneStart behind, less_behind;
  behind.backlog = 10;
  less_behind.backlog = 5;
  plans = Plan(batch, {behind, less_behind, behind});
  EXPECT_TRUE(plans[0].runs.empty());
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  // the backlog only breaks ties: the lane that continues the run gets it
  auto held = Held("a", index, 0);
  held.backlog = 1000;
  plans = Plan(batch, {{}, held});
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  // a lane whose decoder a later run of the batch continues on goes before the backlog
  auto long_index = MakeIndex(1000, 250);
  Batch two;
  two.Add("b", &long_index, 0, 0, 47);
  two.Add("a", &long_index, 1, 500, 515);
  LaneStart spare;
  spare.backlog = 10;
  plans = Plan(two, {Held("a", long_index, 500), spare});
  EXPECT_EQ(plans[0].runs, std::vector<int>({1}));
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
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

TEST(DecodeLanePlannerTest, TieKeepsTheHeldDecoder) {
  // b costs the same on both lanes; it goes to the lane without a decoder, so that a continues on
  // lane 0's decoder instead of reopening its file
  auto index = MakeIndex(1000, 250);
  Batch batch;
  batch.Add("b", &index, 0, 0, 47);
  batch.Add("a", &index, 1, 500, 515);
  auto plans = Plan(batch, {Held("a", index, 500), {}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({1}));
  EXPECT_EQ(plans[0].load, 16);  // no opening, no preroll
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, kOpenCost + 48);
  // without a run of a left in the batch, the tie goes to the lowest lane
  Batch only_b;
  only_b.Add("b", &index, 0, 0, 47);
  plans = Plan(only_b, {Held("a", index, 500), {}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({0}));
  EXPECT_TRUE(plans[1].runs.empty());
}

TEST(DecodeLanePlannerTest, RunWithoutFrames) {
  // A sample whose frames are all constant padding decodes no frame, but the reader still opens
  // its file: an opening on a lane without the file, nothing on a lane that has it
  auto index = MakeIndex(10, 10);
  auto no_frames = Frames({}, 10, 13, 1, boundary::BoundaryType::CONSTANT, 0, 10, 10);
  ASSERT_TRUE(no_frames.empty());
  Batch batch;
  batch.Add("a", &index, 0, std::vector<std::vector<int>>{no_frames});
  auto plans = Plan(batch, {{}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[0].load, kOpenCost);
  EXPECT_EQ(plans[0].pictures, 0);
  plans = Plan(batch, {{}, Held("a", index, 5)});
  EXPECT_TRUE(plans[0].runs.empty());
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, 0);
  EXPECT_EQ(plans[1].pictures, 0);
}

TEST(DecodeLanePlannerTest, InvalidInput) {
  auto index = MakeIndex(50, 10);
  {  // a run without an index
    Batch batch;
    batch.Add("a", nullptr, 0, 20, 39);
    EXPECT_THROW(Plan(batch, {{}}), std::invalid_argument);
  }
  {  // a frame outside of the index
    Batch batch;
    batch.Add("a", &index, 0, 45, 50);
    EXPECT_THROW(Plan(batch, {{}}), std::out_of_range);
  }
  {  // runs, but no lanes
    Batch batch;
    batch.Add("a", &index, 0, 0, 9);
    EXPECT_THROW(Plan(batch, {}), std::invalid_argument);
  }
  // no runs, no lanes
  EXPECT_TRUE(Plan(Batch(), {}).empty());
}

TEST(DecodeLanePlannerTest, EmptyFilenameIsNotHeld) {
  // A lane without a decoder (an empty filename) does not hold a file with an empty name. Its
  // position is where a held decoder would continue without an opening, so that holding the file
  // would cost less
  auto index = MakeIndex(50, 50);
  Batch batch;
  batch.Add("", &index, 0, 0, 9);
  auto plans = Plan(batch, {{"", {0, -1}}});
  EXPECT_EQ(plans[0].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[0].load, kOpenCost + 10);
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
  std::string gops = make_string(testing::dali_extra_path(), "/db/video/vfr/test_2_vp9.mp4");
  std::string single_gop = make_string(testing::dali_extra_path(), "/db/video/cfr/test_1_vp9.mp4");
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
    ASSERT_NO_FATAL_FAILURE(CheckPlannedPictures({{gops, n - 6, n + 2, 1},
                                                  {gops, 30, 46, 2},
                                                  {gops, 0, 0, 1, {0, n / 3, 2 * n / 3, n - 1}},
                                                  {single_gop, 0, 16, 1},
                                                  {single_gop, 25, 41, 1}},
                                                 lanes, indices, buffer, what + "1"));
    ASSERT_NO_FATAL_FAILURE(CheckPlannedPictures({{gops, 30, 46, 1},
                                                  {gops, 5, 21, 1},
                                                  {single_gop, 16, 32, 1},
                                                  {single_gop, 34, 50, 1}},
                                                 lanes, indices, buffer, what + "2"));
    // regions of the files (a file list), padded past their end
    ASSERT_NO_FATAL_FAILURE(CheckPlannedPictures(
        {{gops, 12, 28, 1, {}, boundary::BoundaryType::REFLECT_101, 12, 20},
         {gops, 38, 50, 1, {}, boundary::BoundaryType::CLAMP, 35, 45},
         {gops, 50, 60, 1, {}, boundary::BoundaryType::CONSTANT, 50, 55},
         {single_gop, 32, 48, 2, {}, boundary::BoundaryType::REFLECT_1001, 32, 40},
         {single_gop, 0, 0, 1, {3, 6, 9}, boundary::BoundaryType::CLAMP, 2, 12}},
        lanes, indices, buffer, what + "3"));
  }
}

}  // namespace dali
