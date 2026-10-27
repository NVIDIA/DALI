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
#include "dali/core/boundary.h"
#include "dali/core/small_vector.h"
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
  // The frames that each sample decodes, in the order its decoder seeks to them: in the range of
  // the file, sorted by frame index, duplicates kept, no constant padding
  SmallVector<span<const int>, 4> samples;
};

/**
 * @brief Where a lane's decoder is when the batch starts.
 */
struct LaneStart {
  std::string_view filename;  // empty: no decoder
  // The frame it returns next (as FramesDecoderBase::NextFrameIdx) and, with an index, the last
  // picture it decoded (see FramesDecoderBase::PositionAt)
  FramesDecoderBase::DecodePosition pos = {-1, -1};
};

/**
 * @brief The runs a lane decodes, in the order it decodes them, and their estimated work.
 */
struct LanePlan {
  std::vector<int> runs;  // indices into the planned runs
  int64_t load = 0;
  int64_t pictures = 0;  // the part of `load` that is pictures decoded (not openings or resets)
};

/**
 * @brief The frames a sample decodes, in the order its decoder seeks to them (see
 * FramesDecoderBase::DecodeFrames): in the range of the file, sorted, duplicates kept, no constant
 * padding.
 *
 * @param frame_idxs     the frames of the sample; if empty, from `start` to `end` (exclusive) by
 *                       `stride`
 * @param boundary_type  the padding of the frames outside of [roi_start, roi_end)
 * @param frames         the frames (cleared first)
 */
DLL_PUBLIC void SampleFrames(span<const int> frame_idxs, int start, int end, int stride,
                             boundary::BoundaryType boundary_type, int roi_start, int roi_end,
                             SmallVector<int, 32> &frames);

/**
 * @brief Whether a decoder whose next frame is `next` decodes through to `frame` without
 * seeking (see FramesDecoderBase::SeekFrame).
 */
DLL_PUBLIC bool CanReachWithoutSeek(const FrameIndex *index, int next, int frame);

/**
 * @brief Assigns the runs to lanes: longest first, each to the lane that would finish it first.
 *
 * A lane decodes the runs of one file together, by first frame: first the file its decoder holds
 * (starting with the run that continues where the decoder is), then the others in batch order,
 * each from a newly opened decoder.
 *
 * The work is counted in pictures decoded (see FramesDecoderBase::AddDecodeWork), simulated in
 * that order: B-frames, hidden lead-ins, seeks and resets, plus `open_cost` per opening or reset.
 * Without an index, a run costs its frame range. With one, the `pictures` of a lane's plan are
 * the pictures its decoders decode (see FramesDecoderBase::DecodedPictures).
 *
 * @param runs       the runs of the batch
 * @param lanes      where the lanes' decoders are
 * @param open_cost  the work of opening a decoder, in pictures
 * @param plans      the plan of every lane (resized to the number of lanes)
 */
DLL_PUBLIC void PlanLanes(span<const PlannedRun> runs, span<const LaneStart> lanes,
                          int64_t open_cost, std::vector<LanePlan> &plans);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_DECODE_LANE_PLANNER_H_
