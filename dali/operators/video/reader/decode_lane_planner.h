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

#ifndef DALI_OPERATORS_VIDEO_READER_DECODE_LANE_PLANNER_H_
#define DALI_OPERATORS_VIDEO_READER_DECODE_LANE_PLANNER_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "dali/core/api_helper.h"
#include "dali/core/span.h"
#include "dali/operators/video/frames_decoder_base.h"

namespace dali {

/**
 * @brief Samples of one file that one decoder decodes in a row, as the planner sees them.
 */
struct PlannedRun {
  std::string_view filename;
  const FrameIndex *index;  // null: no index, the work is estimated from the frame range
  int batch_idx;            // the smallest position of its samples in the batch
  int first, last;          // the frames decoded, from first to last; -1 if none
};

/**
 * @brief Where a lane's decoder is when the batch starts.
 */
struct LaneStart {
  std::string_view filename;  // empty: no decoder
  int next_frame = -1;        // as FramesDecoderBase::NextFrameIdx
};

/**
 * @brief The runs a lane decodes, in the order it decodes them, and their estimated work.
 */
struct LanePlan {
  std::vector<int> runs;  // indices into the planned runs
  int64_t load = 0;
};

/**
 * @brief Whether a decoder whose next frame is `next` decodes through to `frame` without
 * seeking (see FramesDecoderBase::SeekFrame).
 */
DLL_PUBLIC bool CanReachWithoutSeek(const FrameIndex *index, int next, int frame);

/**
 * @brief Assigns the runs to lanes: longest first, each to the lane that would finish it first,
 * counting what the lane's decoder saves (no reopening, no seeking).
 *
 * A lane then decodes the runs of one file together, by first frame: first the file its decoder
 * holds (starting with the run that continues where the decoder is), then the others in batch
 * order.
 *
 * @param runs       the runs of the batch
 * @param lanes      where the lanes' decoders are
 * @param open_cost  the work of opening a decoder, in frames
 * @param plans      the plan of every lane (resized to the number of lanes)
 */
DLL_PUBLIC void PlanLanes(span<const PlannedRun> runs, span<const LaneStart> lanes,
                          int64_t open_cost, std::vector<LanePlan> &plans);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_DECODE_LANE_PLANNER_H_
