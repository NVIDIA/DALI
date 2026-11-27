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
 *
 * The filename, the index and the frames of the samples are not owned: they must outlive the call
 * to PlanLanes.
 */
struct PlannedRun {
  std::string_view filename;
  const FrameIndex *index;  // the index of the file; not null
  int batch_idx;            // the smallest position of its samples in the batch
  // The frames that each sample decodes, in the order its decoder seeks to them (see
  // SampleFrames): in [0, index->size()), sorted, duplicates kept, no constant padding
  SmallVector<span<const int>, 4> samples;
};

/**
 * @brief Where a lane's decoder is when the batch starts.
 *
 * The filename is not owned: it must outlive the call to PlanLanes.
 */
struct LaneStart {
  std::string_view filename;  // empty: no decoder
  // The frame it returns next (as FramesDecoderBase::NextFrameIdx) and, with an index, the last
  // picture it decoded (see FramesDecoderBase::PositionAt)
  FramesDecoderBase::DecodePosition pos = {-1, -1};
  // The work of earlier batches that the lane may still be doing when the batch starts, beyond
  // that of the lane with the least
  int64_t backlog = 0;
};

/**
 * @brief The runs a lane decodes, in the order it decodes them, and their estimated work.
 */
struct LanePlan {
  std::vector<int> runs;  // indices into the planned runs
  // The estimated work, in pictures (see PlanLanes); it treats all pictures as equally costly
  int64_t load = 0;
  // The part of `load` that is pictures (not openings or resets): an estimate of the pictures the
  // decoder goes through (see FramesDecoderBase::DecodeWork). It is exactly the CPU decoder's
  // DecodedPictures when the lanes' decoders are where LaneStart says; the GPU decoder leaves out
  // the H.264 non-reference pictures it skips, and counts a VP9 superframe or a field pair as more
  // than one picture.
  int64_t pictures = 0;
};

/**
 * @brief The frames of a sample, in the sample's order, padded as FramesDecoderBase::HandleBoundary
 * pads them to the region of the file [roi_start, roi_end): the frames the reader passes to
 * FramesDecoderBase::DecodeFrames, and the sample's frame indices. A constant frame is -1.
 *
 * @param frames         the frames (cleared first)
 * @param frame_idxs     the frames of the sample; if empty, from `start` to `end` (exclusive) by
 *                       `stride`, which must be positive
 * @param boundary_type  the padding of the frames outside of the region
 */
DLL_PUBLIC void RegionFrames(SmallVector<int, 32> &frames, span<const int> frame_idxs, int start,
                             int end, int stride, boundary::BoundaryType boundary_type,
                             int roi_start, int roi_end);

/**
 * @brief The frames a sample decodes, in the order its decoder seeks to them: in the range of the
 * file, sorted, duplicates kept, no constant padding.
 *
 * These are the frames of RegionFrames, padded to the file [0, num_frames) as
 * FramesDecoderBase::DecodeFrames does, without the constant frames, and sorted. The second
 * padding changes nothing when the region is within the file.
 *
 * @param frames         the frames (cleared first)
 * @param frame_idxs     the frames of the sample; if empty, from `start` to `end` (exclusive) by
 *                       `stride`, which must be positive
 * @param boundary_type  the padding of the frames outside of the region, or of the file
 * @param num_frames     the number of frames of the file
 */
DLL_PUBLIC void SampleFrames(SmallVector<int, 32> &frames, span<const int> frame_idxs, int start,
                             int end, int stride, boundary::BoundaryType boundary_type,
                             int roi_start, int roi_end, int num_frames);

/**
 * @brief Whether a decoder of a file with this index, whose next frame is `next`, decodes through
 * to `frame` without seeking (see FramesDecoderBase::SeekFrame).
 */
DLL_PUBLIC bool CanReachWithoutSeek(const FrameIndex &index, int next, int frame);

/**
 * @brief The number of decode lanes picked automatically for a batch of `batch_size` samples: up
 * to `max_lanes`, then as few as decode a full batch in as many rounds.
 *
 * A lane decodes its samples one after another, so a batch takes as many rounds as the lane with
 * the most samples has: ceil(batch_size / lanes). 16 samples on 7 lanes take 3 rounds, as on 6
 * lanes, which use less memory; 16 samples on 14 lanes take 2 rounds, as on 8 lanes.
 *
 * @param max_lanes   the most lanes to use (e.g. num_threads, capped per NVDEC engine)
 * @param batch_size  the number of samples in a batch
 * @return            at least 1, at most min(max_lanes, batch_size)
 */
DLL_PUBLIC int EvenSplitLanes(int max_lanes, int batch_size);

/**
 * @brief Assigns the runs to lanes: longest first, each to the lane that would finish it first,
 * counting the work of earlier batches that the lane may still be doing (LaneStart::backlog). So a
 * run that continues where a busy lane's decoder is goes to another lane when opening a decoder
 * there costs less than waiting for the busy one. On a tie, it prefers a lane whose decoder no run
 * left in the batch needs, then the lane with the least backlog, then the lowest lane.
 *
 * A lane decodes the runs of one file together, by first frame: first the file its decoder holds
 * (starting with the run that continues where the decoder is), then the others in batch order,
 * each from a newly opened decoder.
 *
 * The work is counted in pictures decoded (see FramesDecoderBase::AddDecodeWork), simulated from
 * the files' indices in that order: B-frames, hidden lead-ins, seeks and resets, plus `open_cost`
 * per opening or reset. Every picture counts the same, whatever its size, so the lanes' work is
 * balanced in time only as far as the files' pictures take about as long to decode.
 *
 * A run is never split between lanes. So a batch that is one run (e.g. consecutive samples of one
 * long video, read in order) is decoded by one lane, however long it is, while the other lanes
 * have nothing to do.
 *
 * @param plans      the plan of every lane (resized to the number of lanes)
 * @param runs       the runs of the batch; each with an index, and its frames within it
 * @param lanes      where the lanes' decoders are; at least one lane if there are runs
 * @param open_cost  the work of opening a decoder, in pictures
 */
DLL_PUBLIC void PlanLanes(std::vector<LanePlan> &plans, span<const PlannedRun> runs,
                          span<const LaneStart> lanes, int64_t open_cost);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_DECODE_LANE_PLANNER_H_
