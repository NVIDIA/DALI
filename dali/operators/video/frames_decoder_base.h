// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
}

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

namespace dali {

struct IndexEntry {
  int64_t pts;
  int last_keyframe_id;
  bool is_keyframe;
  bool is_flush_frame;
};

struct FrameIndex {
  std::string filename;
  std::vector<IndexEntry> index;
  AVRational timebase;

  size_t size() const {
    return index.size();
  }

  IndexEntry& operator[](size_t idx) {
    assert(idx < index.size());
    return index[idx];
  }

  const IndexEntry& operator[](size_t idx) const {
    assert(idx < index.size());
    return index[idx];
  }

  /**
   * @brief Returns the index of the frame that has the given timestamp
   *
   * @param timestamp Timestamp of the frame to seek to
   * @param inclusive If true, the seek will be to a frame that has this timestamp or a previous one
   */
  int GetFrameIdxByTimestamp(int64_t timestamp, bool inclusive = false) const {
    int frame_idx = 0;
    for (size_t i = 1; i < index.size(); i++) {
      if (index[i].pts >= timestamp) {
        frame_idx = i;
        break;
      }
    }
    if (inclusive) {
      if (frame_idx > 0 && index[frame_idx - 1].pts <= timestamp) {
        frame_idx--;
      }
      assert(index[frame_idx].pts <= timestamp);
    }
    return frame_idx;
  }
};

/**
 * @brief Helper representing video file kept in memory. Allows reading and seeking.
 */
struct MemoryVideoFile {
  MemoryVideoFile(const char *data, int64_t size)
    : data_(data), size_(size), position_(0) {}

  int Read(unsigned char *buffer, int buffer_size);

  int64_t Seek(int64_t new_position, int origin);

  const char *data_;
  const int64_t size_;
  int64_t position_;
};

static void DestroyAvObject(AVIOContext **ctx) {
  assert(*ctx != nullptr);
  uint8_t *buffer = (*ctx)->buffer;
  avio_context_free(ctx);
  if (buffer) {
    av_freep(&buffer);
  }
}

static void DestroyAvObject(AVFormatContext **ctx) {
  assert(*ctx != nullptr);
  auto custom_dealloc = (*ctx)->flags & AVFMT_FLAG_CUSTOM_IO;
  auto avio_ctx = (*ctx)->pb;
  avformat_close_input(ctx);
  if (custom_dealloc && avio_ctx) {
    DestroyAvObject(&avio_ctx);
  }
}

static void DestroyAvObject(AVCodecContext **ctx) {
  assert(*ctx != nullptr);
  avcodec_free_context(ctx);
}

static void DestroyAvObject(AVFrame **frame) {
  assert(*frame != nullptr);
  av_frame_free(frame);
}

static void DestroyAvObject(AVPacket **packet) {
  assert(*packet != nullptr);
  av_packet_free(packet);
}

template <typename T>
class AVUniquePtr : public UniqueHandle<T *, AVUniquePtr<T>> {
 public:
  using Base = UniqueHandle<T *, AVUniquePtr<T>>;
  using Base::Base;
  using Base::handle_;

  static void DestroyHandle(T *&handle) {
    assert(handle != nullptr);
    DestroyAvObject(&handle);
  }
  static constexpr T *null_handle() { return nullptr; }

  T *operator->() const { return Base::get(); }
  T &operator*() const { return *Base::get(); }
  T **operator&() { return &handle_; }  // NOLINT(runtime/operator)
};

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
  explicit FramesDecoderBase(const std::string &filename);

  /**
   * @brief Initialize the decoder from a memory buffer.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param source_info Source information for the video file.
   */
  explicit FramesDecoderBase(const char *memory_file, size_t memory_file_size,
                             std::string_view source_info = {});

  /**
   * @brief Number of frames in the video. It returns 0, if this information is unavailable.
   */
  int64_t NumFrames();

  /**
   * @brief Set the number of frames in the video, which can be used to avoid parsing the file.
   */
  void SetNumFrames(int64_t num_frames) {
    num_frames_ = num_frames;
  }

  /**
   * @brief Width of a video frame in pixels
   */
  int Width() const {
    return codec_params_->width;
  }

  /**
   * @brief Height of a video frame in pixels
   */
  int Height() const {
    return codec_params_->height;
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
    return is_vfr_;
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
    return ctx_->streams[stream_id_]->time_base;
  }

  /**
   * @brief Returns the index of the frame that has the given timestamp
   *
   * @param timestamp Timestamp of the frame to seek to
   * @param inclusive If true, the seek will be to a frame that has this timestamp or a previous one
   */
  int GetFrameIdxByTimestamp(int64_t timestamp, bool inclusive = false) const {
    DALI_ENFORCE(HasIndex(), "No index available, cannot seek by timestamp");
    return Index().GetFrameIdxByTimestamp(timestamp, inclusive);
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
  bool HasIndex() const { return Index().size() > 0; }

  /**
   * @brief Builds the index of the video file.
   */
  virtual void BuildIndex();

  virtual ~FramesDecoderBase() = default;
  FramesDecoderBase(FramesDecoderBase&&) = default;
  FramesDecoderBase& operator=(FramesDecoderBase&&) = default;

  std::string Filename() {
    return filename_.size() ? filename_ : "memory file";
  }

  bool IsValid() {
    return is_valid_;
  }

  const FrameIndex& Index() const {
    return index_;
  }

  void SetOutputType(DALIDataType dtype) {
    dtype_ = dtype;
  }

  void SetImageType(DALIImageType image_type) {
    image_type_ = image_type;
  }

 protected:
  /**
   * @brief Select a stream to decode. If stream_id is -1, the video stream will be selected automatically.
   */
  virtual bool SelectVideoStream(int stream_id = -1);

  AVUniquePtr<AVFormatContext> ctx_;
  AVUniquePtr<AVCodecContext> codec_ctx_;
  AVUniquePtr<AVFrame> frame_;
  AVUniquePtr<AVPacket> packet_;
  AVCodecParameters *codec_params_ = nullptr;
  int stream_id_ = -1;
  bool is_file_open_ = false;

  int OpenFile(const std::string& filename);
  int OpenMemoryFile(MemoryVideoFile& memory_video_file);

  FrameIndex index_;

  int next_frame_idx_ = 0;

  DALIImageType image_type_ = DALI_RGB;
  DALIDataType dtype_ = DALI_UINT8;
  VideoColorSpaceConversionType conversion_type_ = VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB;

  // False when the file doesn't have any correct content or doesn't have a valid video stream
  bool is_valid_ = false;

 private:
  void DetectVariableFrameRate();

  bool CheckDimensions();

  void ParseNumFrames();

  bool IsFormatSeekable();

  std::string GetAllStreamInfo() const;

  int channels_ = 3;
  bool is_vfr_ = false;

  std::string filename_ = {};
  std::unique_ptr<MemoryVideoFile> memory_video_file_;

  int num_frames_ = -1;
  bool can_seek_ = true;  // at first, we assume that the video is seekable
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_
