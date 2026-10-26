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
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "dali/operators/video/reader/decode_lane_planner.h"

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

std::vector<LanePlan> Plan(const std::vector<PlannedRun> &runs,
                           const std::vector<LaneStart> &lanes) {
  std::vector<LanePlan> plans;
  PlanLanes(make_cspan(runs), make_cspan(lanes), kOpenCost, plans);
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
           CanReachWithoutSeek(run.index, start.next_frame, run.first);
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

}  // namespace

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
  std::vector<PlannedRun> runs = {
    {"b", &index, 1, 0, 19},
    {"a", &index, 0, 0, 29},
    {"c", &index, 2, 0, 9},
  };
  auto plans = Plan(runs, {{}, {}});
  // a: lane 0 (a tie, to the lowest lane), b: lane 1, c: lane 1 (36 + 26 < 46 + 26)
  EXPECT_EQ(plans[0].runs, std::vector<int>({1}));
  EXPECT_EQ(plans[0].load, 30 + kOpenCost);
  EXPECT_EQ(plans[1].runs, std::vector<int>({0, 2}));
  EXPECT_EQ(plans[1].load, 20 + 10 + 2 * kOpenCost);
}

TEST(DecodeLanePlannerTest, HeldFileContinues) {
  auto index = MakeIndex(100, 50);
  // frames 20-39 of the GOP from frame 0: 20 frames of preroll after a seek
  std::vector<PlannedRun> runs = {{"a", &index, 0, 20, 39}};
  auto plans = Plan(runs, {{"b", 0}, {"a", 20}});
  EXPECT_TRUE(plans[0].runs.empty());
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, 20);  // no opening, no preroll
  // a decoder that has to seek back pays the preroll, but not the opening
  plans = Plan(runs, {{"b", 0}, {"a", 45}});
  EXPECT_EQ(plans[1].runs, std::vector<int>({0}));
  EXPECT_EQ(plans[1].load, 20 + 20);
}

TEST(DecodeLanePlannerTest, NoIndex) {
  // without an index, a run costs its frames (and the opening), with no preroll
  std::vector<PlannedRun> runs = {{"a", nullptr, 0, 20, 39}};
  auto plans = Plan(runs, {{}});
  EXPECT_EQ(plans[0].load, 20 + kOpenCost);
}

TEST(DecodeLanePlannerTest, LaneOrder) {
  // Random batches: runs of a few files with distinct first frames (no ties in the order)
  std::mt19937 rng(1234);
  auto index = MakeIndex(300, 30);
  std::vector<std::string> files = {"a", "b", "c", "d"};
  for (int iter = 0; iter < 200; iter++) {
    std::vector<PlannedRun> runs;
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
      runs.push_back({file, &index, batch_idx[r], first, first + static_cast<int>(rng() % 20)});
    }
    std::vector<LaneStart> lanes(std::uniform_int_distribution<>(1, 4)(rng));
    for (auto &lane : lanes) {
      if (rng() % 3)
        lane = {files[rng() % files.size()], static_cast<int>(rng() % 300)};
    }
    auto plans = Plan(runs, lanes);
    std::vector<int> planned;
    for (size_t l = 0; l < lanes.size(); l++) {
      EXPECT_EQ(plans[l].runs, ReferenceOrder(runs, lanes[l], plans[l].runs))
          << "iteration " << iter << ", lane " << l;
      planned.insert(planned.end(), plans[l].runs.begin(), plans[l].runs.end());
    }
    std::sort(planned.begin(), planned.end());
    ASSERT_EQ(planned.size(), runs.size());
    for (int r = 0; r < num_runs; r++)
      EXPECT_EQ(planned[r], r);
  }
}

}  // namespace dali
