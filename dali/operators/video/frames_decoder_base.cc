// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/frames_decoder_base.h"
#include <memory>
#include "dali/core/error_handling.h"
#include "dali/core/small_vector.h"
#include "dali/operators/video/video_utils.h"

namespace dali {

namespace {

DALIImageType CheckImageType(DALIImageType image_type) {
  DALI_ENFORCE(image_type == DALI_YCbCr || image_type == DALI_RGB,
               make_string("Invalid image type: ", image_type));
  return image_type;
}

}  // namespace

int64_t FramesDecoderBase::NumFrames() {
  if (!demuxer_.NumFramesKnown()) {
    demuxer_.ParseNumFrames();
    Reset();
  }
  return demuxer_.NumFrames();
}

bool FramesDecoderBase::SelectVideoStream(int stream_id) {
  if (!demuxer_.SelectVideoStream(stream_id)) {
    return false;
  }
  next_frame_idx_ = 0;
  return true;
}

FramesDecoderBase::FramesDecoderBase(const std::string &filename, DALIImageType image_type)
    : image_type_(CheckImageType(image_type)), demuxer_(filename) {
  is_valid_ = demuxer_.IsValid();
  next_frame_idx_ = 0;
}

FramesDecoderBase::FramesDecoderBase(const char *memory_file, size_t memory_file_size,
                                     std::string_view source_info, DALIImageType image_type)
    : image_type_(CheckImageType(image_type)),
      demuxer_(memory_file, memory_file_size, source_info) {
  is_valid_ = demuxer_.IsValid();
  next_frame_idx_ = 0;
}

void FramesDecoderBase::BuildIndex() {
  if (HasIndex()) {
    return;
  }
  demuxer_.BuildIndex();
  Reset();
}

bool FramesDecoderBase::AvSeekFrame(int64_t timestamp, int frame_id) {
  if (!demuxer_.Seek(timestamp))
    return false;

  LOG_LINE << "Seeked to frame " << frame_id << std::endl;
  Flush();

  next_frame_idx_ = frame_id;
  return true;
}

void FramesDecoderBase::Reset() {
  LOG_LINE << "Reset: Reopening stream." << std::endl;
  int stream_id = demuxer_.StreamId();

  demuxer_.Reopen();

  is_valid_ = true;
  next_frame_idx_ = 0;

  SelectVideoStream(stream_id);
}

void FramesDecoderBase::SeekFrame(int frame_id) {
  LOG_LINE << "SeekFrame: Seeking to frame " << frame_id
            << " (current=" << next_frame_idx_ << ")" << std::endl;

  // TODO(awolant): Optimize seeking:
  //  - for CFR, when we know pts, but don't know keyframes
  DALI_ENFORCE(
      frame_id >= 0 && frame_id < NumFrames(),
      make_string("Invalid seek frame id. frame_id = ", frame_id, ", num_frames = ", NumFrames()));

  if (frame_id == next_frame_idx_) {
    LOG_LINE << "Already at requested frame" << std::endl;
    return;  // No need to seek
  }

  if (next_frame_idx_ < 0 || (HasIndex() && next_frame_idx_ >= NumFrames())) {
    LOG_LINE << "Resetting decoder because next_frame_idx_ is out of bounds" << std::endl;
    Reset();
  }
  assert(next_frame_idx_ >= 0);

  // If we are seeking to a frame that is before the current frame,
  // or we are seeking to a frame that is more than MINIMUM_SEEK_LEAP frames away,
  // or the current frame index is invalid (e.g. end of file),
  // we will to seek to the nearest keyframe first
  LOG_LINE << "SeekFrame: frame_id=" << frame_id << ", next_frame_idx=" << next_frame_idx_
           << std::endl;
  constexpr int MINIMUM_SEEK_LEAP = 10;
  if (frame_id < next_frame_idx_ || frame_id > next_frame_idx_ + MINIMUM_SEEK_LEAP) {
    // If we have an index, we can seek to the nearest keyframe first
    if (HasIndex()) {
      LOG_LINE << "Using index to find nearest keyframe" << std::endl;
      const auto &current_frame = demuxer_.GetIndex()[next_frame_idx_];
      const auto &requested_frame = demuxer_.GetIndex()[frame_id];
      auto keyframe_id = requested_frame.last_keyframe_id;
      // if we are seeking to a different keyframe than the current frame,
      // or if we are seeking to a frame that is before the current frame,
      // we need to seek to the keyframe first
      LOG_LINE << "current_frame.last_keyframe_id=" << current_frame.last_keyframe_id
               << ", keyframe_id=" << keyframe_id << ", frame_id=" << frame_id
               << ", next_frame_idx_=" << next_frame_idx_ << std::endl;

      if (current_frame.last_keyframe_id != keyframe_id || frame_id < next_frame_idx_) {
        // We are seeking to a different keyframe than the current frame,
        // so we need to seek to the keyframe first
        auto &keyframe_entry = demuxer_.GetIndex()[keyframe_id];
        LOG_LINE << "Seeking to key frame " << keyframe_id << " timestamp " << keyframe_entry.pts
                 << " for requested frame " << frame_id << " timestamp " << requested_frame.pts
                 << std::endl;

        if (!AvSeekFrame(keyframe_entry.pts, keyframe_id)) {
          LOG_LINE << "Failed to seek to keyframe " << keyframe_id << " timestamp "
                   << keyframe_entry.pts << ". Resetting decoder." << std::endl;
          Reset();
        }
      }
    } else if (frame_id < next_frame_idx_) {
      LOG_LINE << "No index & seeking backwards. Resetting decoder." << std::endl;
      Reset();
    }
  }
  LOG_LINE << "After seeking: next_frame_idx_=" << next_frame_idx_ << ", frame_id=" << frame_id
           << std::endl;
  assert(next_frame_idx_ <= frame_id);
  // Skip all remaining frames until the requested frame
  LOG_LINE << "Skipping frames from " << next_frame_idx_ << " to " << frame_id << std::endl;
  for (int i = next_frame_idx_; i < frame_id; i++) {
    ReadNextFrame(nullptr);
  }
  LOG_LINE << "After skipping: next_frame_idx_=" << next_frame_idx_ << ", frame_id=" << frame_id
           << std::endl;
  assert(next_frame_idx_ == frame_id);
}

int FramesDecoderBase::HandleBoundary(boundary::BoundaryType boundary_type, int frame_id, int roi_start, int roi_end) {
  DALI_ENFORCE(boundary_type == boundary::BoundaryType::CLAMP ||
                   boundary_type == boundary::BoundaryType::CONSTANT ||
                   boundary_type == boundary::BoundaryType::REFLECT_1001 ||
                   boundary_type == boundary::BoundaryType::REFLECT_101 ||
                   boundary_type == boundary::BoundaryType::ISOLATED,
               make_string("Invalid boundary type: ", boundary::to_string(boundary_type)));
  if (frame_id >= roi_start && frame_id < roi_end) {
    return frame_id;
  }
  switch (boundary_type) {
    case boundary::BoundaryType::CLAMP:
      return std::clamp(frame_id, roi_start, roi_end - 1);
    case boundary::BoundaryType::CONSTANT:
      return -1;
    case boundary::BoundaryType::REFLECT_1001:
      return boundary::idx_reflect_1001(frame_id, roi_end);
    case boundary::BoundaryType::REFLECT_101:
      return boundary::idx_reflect_101(frame_id, roi_end);
    case boundary::BoundaryType::ISOLATED:
    default:
      DALI_FAIL(make_string(
          "Unexpected out-of-bounds frame index ", frame_id,
          " for pad_mode = 'none' and a sample containing a ROI with ", roi_end - roi_start,
          " frames. Range of valid frame indices for this sample is [", roi_start, ", ", roi_end,
          "). Change `pad_mode` to other than 'none' "
          "for out-of-bounds sampling."));
  }
}

void FramesDecoderBase::DecodeFramesImpl(uint8_t *data,
                                         SmallVector<std::pair<int, int>, 32> frame_ids,
                                         boundary::BoundaryType boundary_type,
                                         const uint8_t *constant_frame,
                                         span<double> out_timestamps) {
  DALI_ENFORCE(constant_frame != nullptr || boundary_type != boundary::BoundaryType::CONSTANT,
               make_string("Constant frame must be provided if boundary type is CONSTANT"));

  uint8_t *last_out_frame_start = nullptr;
  for (auto &[frame_id, i] : frame_ids) {
    uint8_t* out_frame_start = data + ptrdiff_t(i) * FrameSize();
    assert(out_frame_start >= data);
    if (frame_id >= 0 && frame_id < NumFrames()) {
      LOG_LINE << "Decoding frame " << frame_id << " to position " << i << std::endl;
      SeekFrame(frame_id);
      ReadNextFrame(out_frame_start);
      last_out_frame_start = out_frame_start;
    } else if (frame_id < 0) {
      LOG_LINE << "Copying constant frame to position " << i << std::endl;
      CopyFrame(out_frame_start, constant_frame);
    } else {
      LOG_LINE << "Copying last decoded frame to position " << i << std::endl;
      CopyFrame(out_frame_start, last_out_frame_start);
    }
  }

  if (!out_timestamps.empty()) {
    LOG_LINE << "Computing timestamps for " << out_timestamps.size() << " frames" << std::endl;
    int64_t pts_0 = demuxer_.GetIndex()[0].pts;
    for (auto& [frame_idx, i] : frame_ids) {
      if (frame_idx >= 0) {
        out_timestamps[i] =
            TimestampToSeconds(GetTimebase(), demuxer_.GetIndex()[frame_idx].pts - pts_0);
      } else {
        out_timestamps[i] = -1.0f;
      }
    }
  }
}

void FramesDecoderBase::DecodeFrames(uint8_t *data, span<const int> frame_ids,
                                     boundary::BoundaryType boundary_type,
                                     const uint8_t *constant_frame,
                                     span<double> out_timestamps) {
  LOG_LINE << "DecodeFrames: " << frame_ids.size() << " frames, boundary_type="
           << boundary::to_string(boundary_type) << std::endl;

  SmallVector<std::pair<int, int>, 32> sorted_frame_ids;
  size_t num_frames = frame_ids.size();
  sorted_frame_ids.reserve(num_frames);
  for (int i = 0; i < static_cast<int>(frame_ids.size()); i++) {
    sorted_frame_ids.push_back({HandleBoundary(boundary_type, frame_ids[i], 0, NumFrames()), i});
  }
  std::sort(sorted_frame_ids.begin(), sorted_frame_ids.end());
  DecodeFramesImpl(data, sorted_frame_ids, boundary_type, constant_frame, out_timestamps);
}


void FramesDecoderBase::DecodeFrames(uint8_t *data, int start_frame, int end_frame, int stride,
                                     boundary::BoundaryType boundary_type,
                                     const uint8_t *constant_frame,
                                     span<double> out_timestamps) {
  LOG_LINE << "DecodeFrames: start=" << start_frame << ", end=" << end_frame
           << ", stride=" << stride << std::endl;

  SmallVector<std::pair<int, int>, 32> sorted_frame_ids;
  size_t num_frames = (end_frame - start_frame + stride - 1) / stride;
  sorted_frame_ids.reserve(num_frames);
  for (int i = 0; i < static_cast<int>(num_frames); i++) {
    sorted_frame_ids.push_back(
        {HandleBoundary(boundary_type, start_frame + i * stride, 0, NumFrames()), i});
  }
  std::sort(sorted_frame_ids.begin(), sorted_frame_ids.end());
  DecodeFramesImpl(data, sorted_frame_ids, boundary_type, constant_frame, out_timestamps);
}

}  // namespace dali
