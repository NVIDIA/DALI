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

#include "dali/operators/video/reader/decode_lane_planner.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace dali {

bool CanReachWithoutSeek(const FrameIndex *index, int next, int frame) {
  if (next < 0 || frame < next)
    return false;
  if (frame - next <= FramesDecoderBase::kMinimumSeekLeap)
    return true;
  return index && frame < static_cast<int>(index->size()) &&
         (*index)[frame].last_keyframe_id == (*index)[next].last_keyframe_id;
}

void PlanLanes(span<const PlannedRun> runs, span<const LaneStart> lanes, int64_t open_cost,
               std::vector<LanePlan> &plans) {
  int num_lanes = lanes.size();
  plans.clear();
  plans.resize(num_lanes);
  std::vector<LaneStart> next(lanes.begin(), lanes.end());  // where the decoders will be
  struct RunCost {
    int run;
    int64_t frames;   // frames decoded, from the first to the last
    int64_t preroll;  // frames decoded before the first one, after a seek
  };
  std::vector<RunCost> costs;
  for (int r = 0; r < static_cast<int>(runs.size()); r++) {
    const auto &run = runs[r];
    int64_t frames = run.first < 0 ? 0 : run.last - run.first + 1;
    int64_t preroll = 0;
    if (run.index && run.first >= 0 && run.first < static_cast<int>(run.index->size()))
      preroll = run.first - (*run.index)[run.first].last_keyframe_id;
    costs.push_back({r, frames, preroll});
  }
  std::stable_sort(costs.begin(), costs.end(), [](const RunCost &a, const RunCost &b) {
    return a.frames + a.preroll > b.frames + b.preroll;
  });
  for (const auto &cost : costs) {
    const auto &run = runs[cost.run];
    int best = 0;
    int64_t best_end = std::numeric_limits<int64_t>::max();
    for (int l = 0; l < num_lanes; l++) {
      int64_t work = cost.frames;
      if (next[l].filename != run.filename)
        work += open_cost + cost.preroll;
      else if (!CanReachWithoutSeek(run.index, next[l].next_frame, run.first))
        work += cost.preroll;
      if (plans[l].load + work < best_end) {
        best = l;
        best_end = plans[l].load + work;
      }
    }
    plans[best].runs.push_back(cost.run);
    plans[best].load = best_end;
    next[best].filename = run.filename;
    next[best].next_frame = run.last < 0 ? next[best].next_frame : run.last + 1;
  }
  for (int l = 0; l < num_lanes; l++) {
    auto &lane_runs = plans[l].runs;
    auto continues = [&](int r) {
      const auto &run = runs[r];
      return run.filename == lanes[l].filename &&
             CanReachWithoutSeek(run.index, lanes[l].next_frame, run.first);
    };
    // files in the order of their first sample in the batch, the decoder's file first
    std::map<std::string_view, int> file_order;
    for (int r : lane_runs) {
      auto [it, inserted] = file_order.emplace(runs[r].filename, runs[r].batch_idx);
      if (!inserted)
        it->second = std::min(it->second, runs[r].batch_idx);
    }
    auto held = file_order.find(lanes[l].filename);
    if (held != file_order.end())
      held->second = -1;
    std::sort(lane_runs.begin(), lane_runs.end(), [&](int a, int b) {
      const auto &ra = runs[a], &rb = runs[b];
      return std::make_tuple(file_order[ra.filename], ra.filename, ra.first) <
             std::make_tuple(file_order[rb.filename], rb.filename, rb.first);
    });
    auto it = std::find_if(lane_runs.begin(), lane_runs.end(), continues);
    if (it != lane_runs.end())
      std::rotate(lane_runs.begin(), it, it + 1);
  }
}

}  // namespace dali
