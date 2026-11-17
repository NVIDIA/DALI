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

bool CanReachWithoutSeek(const FrameIndex &index, int next, int frame) {
  if (next < 0 || frame < next)
    return false;
  if (frame - next <= FramesDecoderBase::kMinimumSeekLeap)
    return true;
  // the decision of SeekFrame and AddDecodeWork
  int size = index.size();
  return next < size && frame < size && FramesDecoderBase::SeekTarget(index, next, frame) < 0;
}

void SampleFrames(span<const int> frame_idxs, int start, int end, int stride,
                  boundary::BoundaryType boundary_type, int roi_start, int roi_end,
                  int num_frames, SmallVector<int, 32> &frames) {
  frames.clear();
  auto add = [&](int frame) {
    // padded to the region of the file, as the reader does, then to the file, as DecodeFrames does
    // (a constant frame, -1 after the first, stays constant after the second)
    int idx = FramesDecoderBase::HandleBoundary(boundary_type, frame, roi_start, roi_end);
    idx = FramesDecoderBase::HandleBoundary(boundary_type, idx, 0, num_frames);
    if (idx >= 0)  // not a constant frame
      frames.push_back(idx);
  };
  if (!frame_idxs.empty()) {
    for (int frame : frame_idxs)
      add(frame);
  } else {
    DALI_ENFORCE(stride > 0, make_string("The stride must be positive, got ", stride, "."));
    for (int frame = start; frame < end; frame += stride)
      add(frame);
  }
  std::sort(frames.begin(), frames.end());
}

FramesDecoderBase::DecodePosition DecoderPosition(FramesDecoderBase &decoder) {
  DALI_ENFORCE(decoder.HasIndex(), "The decoder of a decode lane must have an index.");
  return FramesDecoderBase::PositionAt(decoder.GetIndex(), decoder.NextFrameIdx());
}

namespace {

using DecodePosition = FramesDecoderBase::DecodePosition;
using DecodeWork = FramesDecoderBase::DecodeWork;

// Whether the lane's decoder holds the file (an empty filename means no decoder)
bool Holds(const LaneStart &start, std::string_view filename) {
  return !start.filename.empty() && start.filename == filename;
}

// Whether the decoder of a lane continues to the run without seeking
bool Continues(const PlannedRun &run, const LaneStart &start) {
  return Holds(start, run.filename) &&
         CanReachWithoutSeek(*run.index, start.pos.next_frame, run.first);
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
 * @brief The work of a lane for the runs of one file, decoded in `order`: from the lane's decoder
 * if it holds the file, otherwise from a newly opened one.
 */
DecodeWork FileWork(span<const PlannedRun> runs, const LaneStart &start, span<const int> order) {
  const auto &file = runs[order[0]];
  bool held = Holds(start, file.filename);
  const auto &index = *file.index;
  DecodePosition pos = held ? start.pos : DecodePosition{};
  DecodeWork work;
  work.reopens = held ? 0 : 1;
  for (int r : order) {
    for (auto frames : runs[r].samples) {
      for (int frame : frames)
        FramesDecoderBase::AddDecodeWork(index, pos, frame, work);
    }
  }
  return work;
}

// Checks what PlanLanes expects of its input
void CheckRuns(span<const PlannedRun> runs, int num_lanes) {
  DALI_ENFORCE(runs.empty() || num_lanes > 0, "There must be a decode lane to plan runs on.");
  for (const auto &run : runs) {
    DALI_ENFORCE(run.index, make_string("No frame index for the run of \"", run.filename, "\"."));
    int num_frames = run.index->size();
    for (auto frames : run.samples) {
      for (int frame : frames) {
        DALI_ENFORCE(frame >= 0 && frame < num_frames,
                     make_string("Frame ", frame, " of a run of \"", run.filename,
                                 "\" is out of the range of its index [0, ", num_frames, ")."));
      }
    }
  }
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
  CheckRuns(runs, num_lanes);
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
    LOG_LINE << "PlanLanes: run " << r << " of " << runs[r].filename << " from frame "
             << runs[r].first << ": " << fresh[r] << std::endl;
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
      if (!group && !Holds(lanes[l], run.filename)) {
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
      if (plans[l].load + delta < best_end) {
        best = l;
        best_end = plans[l].load + delta;
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
      return std::make_tuple(Holds(lanes[l], group.filename) ? -1 : batch_idx,
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
