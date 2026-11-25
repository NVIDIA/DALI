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
#include <stdexcept>
#include <tuple>
#include <utility>

#include "dali/core/util.h"

namespace dali {

int EvenSplitLanes(int max_lanes, int batch_size) {
  int lanes = std::max(1, std::min(max_lanes, batch_size));
  if (batch_size <= 0)
    return lanes;
  int rounds = div_ceil(batch_size, lanes);  // samples on the busiest lane
  return div_ceil(batch_size, rounds);       // the fewest lanes that need no more rounds
}

bool CanReachWithoutSeek(const FrameIndex &index, int next, int frame) {
  int size = index.size();
  if (next < 0 || next >= size || frame < next || frame >= size)
    return false;
  // the decision of SeekFrame and AddDecodeWork, the short leap forward included
  return FramesDecoderBase::SeekTarget(index, next, frame) < 0;
}

void RegionFrames(SmallVector<int, 32> &frames, span<const int> frame_idxs, int start, int end,
                  int stride, boundary::BoundaryType boundary_type, int roi_start, int roi_end) {
  frames.clear();
  auto add = [&](int frame) {
    frames.push_back(FramesDecoderBase::HandleBoundary(boundary_type, frame, roi_start, roi_end));
  };
  if (!frame_idxs.empty()) {
    for (int frame : frame_idxs)
      add(frame);
  } else {
    if (stride <= 0)
      throw std::invalid_argument(make_string("The stride must be positive, got ", stride, "."));
    for (int frame = start; frame < end; frame += stride)
      add(frame);
  }
}

void SampleFrames(SmallVector<int, 32> &frames, span<const int> frame_idxs, int start, int end,
                  int stride, boundary::BoundaryType boundary_type, int roi_start, int roi_end,
                  int num_frames) {
  RegionFrames(frames, frame_idxs, start, end, stride, boundary_type, roi_start, roi_end);
  // padded to the file, as DecodeFrames does (a constant frame, -1, stays constant)
  int n = 0;
  for (int frame : frames) {
    int idx = FramesDecoderBase::HandleBoundary(boundary_type, frame, 0, num_frames);
    if (idx >= 0)  // not a constant frame
      frames[n++] = idx;
  }
  frames.resize(n);
  std::sort(frames.begin(), frames.end());
}

FramesDecoderBase::DecodePosition DecoderPosition(const FramesDecoderBase &decoder) {
  if (!decoder.HasIndex())
    throw std::invalid_argument("The decoder of a decode lane must have an index.");
  return FramesDecoderBase::PositionAt(decoder.GetIndex(), decoder.NextFrameIdx());
}

namespace {

using DecodePosition = FramesDecoderBase::DecodePosition;
using DecodeWork = FramesDecoderBase::DecodeWork;

// Whether the lane's decoder holds the file (an empty filename means no decoder)
bool Holds(const LaneStart &start, std::string_view filename) {
  return !start.filename.empty() && start.filename == filename;
}

// The first frame a run decodes: the smallest of its samples' first frames; -1 if none
int FirstFrame(const PlannedRun &run) {
  int first = -1;
  for (auto frames : run.samples) {
    if (!frames.empty() && (first < 0 || frames.front() < first))
      first = frames.front();
  }
  return first;
}

// Whether the decoder of a lane continues to the run, whose first frame is `first`, without
// seeking
bool Continues(const PlannedRun &run, int first, const LaneStart &start) {
  return Holds(start, run.filename) && CanReachWithoutSeek(*run.index, start.pos.next_frame, first);
}

/**
 * @brief Orders the runs of one file on a lane as the lane decodes them: by first frame (then run
 * number), the run that continues from the lane's decoder first.
 */
void OrderFileRuns(span<const PlannedRun> runs, span<const int> first, const LaneStart &start,
                   SmallVector<int, 4> &order) {
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return std::make_pair(first[a], a) < std::make_pair(first[b], b);
  });
  auto it = std::find_if(order.begin(), order.end(),
                         [&](int r) { return Continues(runs[r], first[r], start); });
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
  if (!runs.empty() && num_lanes <= 0)
    throw std::invalid_argument("There must be a decode lane to plan runs on.");
  for (const auto &run : runs) {
    if (!run.index) {
      throw std::invalid_argument(
          make_string("No frame index for the run of \"", run.filename, "\"."));
    }
    int num_frames = run.index->size();
    for (auto frames : run.samples) {
      for (int frame : frames) {
        if (frame < 0 || frame >= num_frames) {
          throw std::out_of_range(
              make_string("Frame ", frame, " of a run of \"", run.filename,
                          "\" is out of the range of its index [0, ", num_frames, ")."));
        }
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

void PlanLanes(std::vector<LanePlan> &plans, span<const PlannedRun> runs,
               span<const LaneStart> lanes, int64_t open_cost) {
  int num_lanes = lanes.size();
  int num_runs = runs.size();
  CheckRuns(runs, num_lanes);
  plans.clear();
  plans.resize(num_lanes);
  // The runs of different files on a lane do not share a decoder: the file the lane's decoder
  // holds goes first, the others are opened anew, one after another. So the work of a lane is the
  // sum of the work of its files, each simulated in the order the lane decodes its runs.
  std::vector<SmallVector<FileGroup, 4>> groups(num_lanes);

  // the first frame of each run, and its work on a newly opened decoder; the longest first
  std::vector<int> first(num_runs);
  std::vector<DecodeWork> fresh_work(num_runs);
  std::vector<int64_t> fresh(num_runs);
  std::vector<int> order(num_runs);
  for (int r = 0; r < num_runs; r++) {
    first[r] = FirstFrame(runs[r]);
    fresh_work[r] = FileWork(runs, LaneStart{}, make_cspan(&r, 1));
    fresh[r] = Cost(fresh_work[r], open_cost);
    order[r] = r;
    LOG_LINE << "PlanLanes: run " << r << " of " << runs[r].filename << " from frame "
             << first[r] << ": " << fresh[r] << std::endl;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return fresh[a] > fresh[b]; });

  // the runs not yet assigned of the file that each lane's decoder holds
  std::vector<int> pending(num_lanes);
  for (int l = 0; l < num_lanes; l++) {
    for (const auto &run : runs)
      pending[l] += Holds(lanes[l], run.filename);
  }

  SmallVector<int, 4> candidate, best_runs;
  for (int r : order) {
    const auto &run = runs[r];
    int best = 0;
    int64_t best_end = std::numeric_limits<int64_t>::max();
    bool best_spare = false;
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
        OrderFileRuns(runs, make_cspan(first), lanes[l], candidate);
        work = FileWork(runs, lanes[l], make_cspan(candidate));
        delta = Cost(work, open_cost) - (group ? Cost(group->work, open_cost) : 0);
      }
      int64_t end = plans[l].load + delta;
      // on a tie, not a lane whose decoder a run left in the batch may continue on
      bool spare = pending[l] == 0 || Holds(lanes[l], run.filename);
      if (end < best_end || (end == best_end && spare && !best_spare)) {
        best = l;
        best_end = end;
        best_spare = spare;
        best_work = work;
        best_runs = candidate;
      }
    }
    for (int l = 0; l < num_lanes; l++) {
      if (Holds(lanes[l], run.filename))
        pending[l]--;
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
