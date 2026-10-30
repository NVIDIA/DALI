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

#ifndef DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_
#define DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavcodec/bsf.h>
}

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "dali/core/boundary.h"
#include "dali/core/common.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/unique_handle.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/video/color_space.h"
#include "dali/operators/video/frames_demuxer.h"

namespace dali {

/**
 * @brief Object representing a video file. Allows access to frames and seeking.
 */
class DLL_PUBLIC FramesDecoderBase {
 public:
  /**
   * @brief Initialize the decoder from a file.
   *
   * @param filename Path to a video file.
   * @param image_type Image type of the video.
   */
  explicit FramesDecoderBase(const std::string &filename, DALIImageType image_type = DALI_RGB);

  /**
   * @brief Initialize the decoder from a memory buffer.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param source_info Source information for the video file.
   */
  explicit FramesDecoderBase(const char *memory_file, size_t memory_file_size,
                             std::string_view source_info = {}, DALIImageType image_type = DALI_RGB);

  /**
   * @brief Number of frames in the video. It returns 0, if this information is unavailable.
   */
  int64_t NumFrames();

  /**
   * @brief Set the number of frames in the video, which can be used to avoid parsing the file.
   */
  void SetNumFrames(int64_t num_frames) {
    demuxer_.SetNumFrames(num_frames);
  }

  /**
   * @brief Width of a video frame in pixels
   */
  int Width() const {
    return demuxer_.Width();
  }

  /**
   * @brief Height of a video frame in pixels
   */
  int Height() const {
    return demuxer_.Height();
  }

  /**
   * @brief Number of channels in a video
   */
  int Channels() const {
      return channels_;
  }

  /**
   * @brief Total number of values in a frame (width * height * channels)
   */
  int FrameSize() const {
    return Channels() * Width() * Height();
  }

  TensorShape<3> FrameShape() const {
    return {Height(), Width(), Channels()};
  }

  TensorShape<4> Shape(int num_frames) const {
    return {num_frames, Height(), Width(), Channels()};
  }

  /**
   * @brief Is video variable frame rate
   */
  bool IsVfr() const {
    return demuxer_.IsVfr();
  }

  /**
   * @brief Reads next frame of the video.
   *
   * @param data Output buffer to copy data to. If nullptr, the frame will be effectively skipped.
   * @return Boolean indicating whether the frame was read or not. False means no more frames in the decoder.
   */
  virtual bool ReadNextFrame(uint8_t *data) = 0;

  /**
   * @brief Seeks to the frame given by id. Next call to ReadNextFrame will return this frame
   *
   * @param frame_id Id of the frame to seek to
   */
  virtual void SeekFrame(int frame_id);

  /**
   * @brief Handles boundary conditions for a given frame index.
   *
   * @param boundary_type The boundary type to apply.
   * @param frame_id The frame index to handle.
   * @param roi_start The start of the region of interest.
   * @param roi_end The end of the region of interest.
   */
  int HandleBoundary(boundary::BoundaryType boundary_type, int frame_id,
                     int roi_start, int roi_end);

  /**
   * @brief Decodes a collection of frames, not necessarily in ascending order, applying a boundary type
   * (what to do when sampling out of bounds).
   * @param data Output buffer to copy data to. Should be of size FrameSize() * frame_ids.size().
   * @param frame_ids Frame indices to decode.
   * @param boundary_type Boundary type to apply
   * @param constant_frame Constant frame data to repeat when sampling out of bounds.
   * @param out_timestamps Output buffer to store timestamps of the decoded frames. If empty, timestamps will not be computed.
   */
  void DecodeFrames(
    uint8_t *data,
    span<const int> frame_ids,
    boundary::BoundaryType boundary_type = boundary::BoundaryType::ISOLATED,
    const uint8_t *constant_frame = nullptr,
    span<double> out_timestamps = {});

  /**
   * @brief Decodes a range of evenly spaced frames, applying a boundary type
   * (what to do when sampling out of bounds).
   * @param data Output buffer to copy data to. Should be of size FrameSize() * frame_ids.size().
   * @param start_frame Start frame index.
   * @param end_frame End frame index.
   * @param stride Stride between frames.
   * @param boundary_type Boundary type to apply
   * @param constant_frame Constant frame data to repeat when sampling out of bounds.
   * @param out_timestamps Output buffer to store timestamps of the decoded frames. If empty, timestamps will not be computed.
   */
  void DecodeFrames(
    uint8_t *data,
    int start_frame,
    int end_frame,
    int stride,
    boundary::BoundaryType boundary_type = boundary::BoundaryType::ISOLATED,
    const uint8_t *constant_frame = nullptr,
    span<double> out_timestamps = {});

  /**
   * @brief Copies a frame from one buffer to another.
   *
   * @param dst Destination buffer.
   * @param src Source buffer.
   */
  virtual void CopyFrame(uint8_t *dst, const uint8_t *src) = 0;

  /**
   * @brief Returns the timebase of the video
   */
  AVRational GetTimebase() const {
    return demuxer_.GetTimebase();
  }

  /**
   * @brief Returns the index of the frame that has the given timestamp
   *
   * @param timestamp Timestamp of the frame to seek to
   * @param inclusive If true, the seek will be to a frame that has this timestamp or a previous one
   */
  int GetFrameIdxByTimestamp(int64_t timestamp, bool inclusive = false) const {
    DALI_ENFORCE(HasIndex(), "No index available, cannot seek by timestamp");
    return demuxer_.GetIndex().GetFrameIdxByTimestamp(timestamp, inclusive);
  }

  /**
   * @brief Seeks to the first frame
   */
  virtual void Reset();

  /**
   * @brief Seeks to the frame given by id using avformat_seek_file
   *
   * @param frame_id Id of the frame to seek to
   *
   * @return Boolean indicating whether the seek was successful
   */
  virtual bool AvSeekFrame(int64_t timestamp, int frame_id);

  /**
   * @brief Flushes the decoder state.
   */
  virtual void Flush() = 0;

  /**
   * @brief Returns index of the frame that will be returned by the next call of ReadNextFrame
   *
   * @return int Index of the next frame to be read
   */
  int NextFrameIdx() { return next_frame_idx_; }

  /**
   * @brief Returns true if the index was built.
   *
   * @return Boolean indicating whether or not the index was created.
   */
  bool HasIndex() const { return demuxer_.HasIndex(); }

  /**
   * @brief Builds the index of the video file.
   */
  virtual void BuildIndex();

  const FrameIndex& GetIndex() const {
    return demuxer_.GetIndex();
  }

  void SetIndex(const FrameIndex& index) {
    demuxer_.SetIndex(index);
  }

  virtual ~FramesDecoderBase() = default;
  FramesDecoderBase(FramesDecoderBase&&) = default;
  FramesDecoderBase& operator=(FramesDecoderBase&&) = default;

  std::string Filename() {
    return demuxer_.Filename();
  }

  bool IsValid() {
    return is_valid_;
  }

  void SetOutputType(DALIDataType dtype) {
    dtype_ = dtype;
  }

 protected:
  void DecodeFramesImpl(uint8_t *data, SmallVector<std::pair<int, int>, 32> frame_ids,
                        boundary::BoundaryType boundary_type,
                        const uint8_t *constant_frame,
                        span<double> out_timestamps);

  /**
   * @brief Select a stream to decode. If stream_id is -1, the video stream will be selected automatically.
   */
  virtual bool SelectVideoStream(int stream_id = -1);

  DALIImageType image_type_ = DALI_RGB;
  DALIDataType dtype_ = DALI_UINT8;

  // Reads the container (the file, the stream, the packets, the index)
  FramesDemuxer demuxer_;

  AVUniquePtr<AVCodecContext> codec_ctx_;
  AVUniquePtr<AVFrame> frame_;
  bool is_file_open_ = false;

  int next_frame_idx_ = 0;

  // False when the file doesn't have any correct content or doesn't have a valid video stream
  bool is_valid_ = false;

 private:
  int channels_ = 3;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_
