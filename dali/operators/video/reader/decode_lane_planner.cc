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

void SampleFrames(span<const int> frame_idxs, int start, int end, int stride,
                  boundary::BoundaryType boundary_type, int roi_start, int roi_end,
                  SmallVector<int, 32> &frames) {
  frames.clear();
  auto add = [&](int frame) {
    int idx = FramesDecoderBase::HandleBoundary(boundary_type, frame, roi_start, roi_end);
    if (idx >= 0)  // not a constant frame
      frames.push_back(idx);
  };
  if (!frame_idxs.empty()) {
    for (int frame : frame_idxs)
      add(frame);
  } else {
    for (int frame = start; frame < end; frame += stride)
      add(frame);
  }
  std::sort(frames.begin(), frames.end());
}

namespace {

using DecodePosition = FramesDecoderBase::DecodePosition;
using DecodeWork = FramesDecoderBase::DecodeWork;

// Whether the decoder of a lane continues to the run without seeking
bool Continues(const PlannedRun &run, const LaneStart &start) {
  return run.filename == start.filename &&
         CanReachWithoutSeek(run.index, start.pos.next_frame, run.first);
}

/**
 * @brief Orders the runs of one file on a lane as the lane decodes them: by first frame (then run
 * number), the run that continues from the lane's decoder first.
 */
void OrderFileRuns(span<const PlannedRun> runs, const LaneStart &start,
                   SmallVector<int, 4> &order) {
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return std::make_pair(runs[a].first, a) < std::make_pair(runs[b].first, b);
  });
  auto it = std::find_if(order.begin(), order.end(),
                         [&](int r) { return Continues(runs[r], start); });
  if (it != order.end())
    std::rotate(order.begin(), it, it + 1);
}

// The work in pictures, an opening or a reset counted as `open_cost`
int64_t Cost(const DecodeWork &work, int64_t open_cost) {
  return work.pictures + open_cost * work.reopens;
}

/**
 * @brief Without an index (or with frames outside of it): the frames of each run, from the first
 * to the last, and the opening of a decoder.
 */
DecodeWork FrameRangeWork(span<const PlannedRun> runs, bool held, span<const int> order) {
  DecodeWork work;
  work.reopens = held ? 0 : 1;
  for (int r : order) {
    if (runs[r].first >= 0)
      work.pictures += runs[r].last - runs[r].first + 1;
  }
  return work;
}

/**
 * @brief The work of a lane for the runs of one file, decoded in `order`: from the lane's decoder
 * if it holds the file, otherwise from a newly opened one.
 */
DecodeWork FileWork(span<const PlannedRun> runs, const LaneStart &start, span<const int> order) {
  const auto &file = runs[order[0]];
  bool held = start.filename == file.filename;
  if (!file.index)
    return FrameRangeWork(runs, held, order);
  const auto &index = *file.index;
  int num_frames = index.size();
  DecodePosition pos = held ? start.pos : DecodePosition{};
  DecodeWork work;
  work.reopens = held ? 0 : 1;
  for (int r : order) {
    for (auto frames : runs[r].samples) {
      for (int frame : frames) {
        if (frame < 0 || frame >= num_frames)
          return FrameRangeWork(runs, held, order);
        FramesDecoderBase::AddDecodeWork(index, pos, frame, work);
      }
    }
  }
  return work;
}

/**
 * @brief The runs of one file on a lane, in the order the lane decodes them, and their work.
 */
struct FileGroup {
  std::string_view filename;
  SmallVector<int, 4> runs;
  DecodeWork work;
};

FileGroup *FindGroup(SmallVector<FileGroup, 4> &groups, std::string_view filename) {
  for (auto &group : groups) {
    if (group.filename == filename)
      return &group;
  }
  return nullptr;
}

}  // namespace

void PlanLanes(span<const PlannedRun> runs, span<const LaneStart> lanes, int64_t open_cost,
               std::vector<LanePlan> &plans) {
  int num_lanes = lanes.size();
  int num_runs = runs.size();
  plans.clear();
  plans.resize(num_lanes);
  // The runs of different files on a lane do not share a decoder: the file the lane's decoder
  // holds goes first, the others are opened anew, one after another. So the work of a lane is the
  // sum of the work of its files, each simulated in the order the lane decodes its runs.
  std::vector<SmallVector<FileGroup, 4>> groups(num_lanes);

  // the work of each run on a newly opened decoder; the longest first
  std::vector<DecodeWork> fresh_work(num_runs);
  std::vector<int64_t> fresh(num_runs);
  std::vector<int> order(num_runs);
  for (int r = 0; r < num_runs; r++) {
    fresh_work[r] = FileWork(runs, LaneStart{}, make_cspan(&r, 1));
    fresh[r] = Cost(fresh_work[r], open_cost);
    order[r] = r;
    LOG_LINE << "PlanLanes: run " << r << " of " << runs[r].filename << " [" << runs[r].first
             << ", " << runs[r].last << "]: " << fresh[r] << std::endl;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return fresh[a] > fresh[b]; });

  SmallVector<int, 4> candidate, best_runs;
  for (int r : order) {
    const auto &run = runs[r];
    int best = 0;
    int64_t best_end = std::numeric_limits<int64_t>::max();
    DecodeWork best_work;
    for (int l = 0; l < num_lanes; l++) {
      FileGroup *group = FindGroup(groups[l], run.filename);
      DecodeWork work;
      int64_t delta;
      candidate.clear();
      if (!group && lanes[l].filename != run.filename) {
        candidate.push_back(r);
        work = fresh_work[r];
        delta = fresh[r];
      } else {
        if (group)
          candidate = group->runs;
        candidate.push_back(r);
        OrderFileRuns(runs, lanes[l], candidate);
        work = FileWork(runs, lanes[l], make_cspan(candidate));
        delta = Cost(work, open_cost) - (group ? Cost(group->work, open_cost) : 0);
      }
      int64_t end = plans[l].load + delta;
      if (end < best_end || (end == best_end && lanes[l].backlog < lanes[best].backlog)) {
        best = l;
        best_end = end;
        best_work = work;
        best_runs = candidate;
      }
    }
    FileGroup *group = FindGroup(groups[best], run.filename);
    if (!group) {
      groups[best].push_back({run.filename, {}, {}});
      group = &groups[best].back();
    }
    group->runs = best_runs;
    group->work = best_work;
    plans[best].load = best_end;
  }

  for (int l = 0; l < num_lanes; l++) {
    // the file the decoder holds first, then the others by their first sample in the batch
    auto file_order = [&](const FileGroup &group) {
      int batch_idx = std::numeric_limits<int>::max();
      for (int r : group.runs)
        batch_idx = std::min(batch_idx, runs[r].batch_idx);
      return std::make_tuple(group.filename == lanes[l].filename ? -1 : batch_idx,
                             group.filename);
    };
    auto &lane_groups = groups[l];
    std::sort(lane_groups.begin(), lane_groups.end(), [&](const FileGroup &a, const FileGroup &b) {
      return file_order(a) < file_order(b);
    });
    for (auto &group : lane_groups) {
      plans[l].runs.insert(plans[l].runs.end(), group.runs.begin(), group.runs.end());
      plans[l].pictures += group.work.pictures;
      LOG_LINE << "PlanLanes: lane " << l << ": " << group.filename << ", " << group.runs.size()
               << " runs: " << group.work.pictures << " pictures, " << group.work.reopens
               << " openings" << std::endl;
    }
    LOG_LINE << "PlanLanes: lane " << l << ": load " << plans[l].load << ", "
             << plans[l].pictures << " pictures" << std::endl;
  }
}

}  // namespace dali
