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
   * @param dtype Data type of the video.
   * @param build_index If set to false index will not be build and some features are unavailable.
   * @param init_codecs If set to false CPU codec part is not initalized, only parser
   */
  explicit FramesDecoderBase(const std::string &filename,
                             DALIImageType image_type = DALI_RGB, DALIDataType dtype = DALI_UINT8,
                             bool build_index = true, bool init_codecs = true);

  /**
   * @brief Initialize the decoder from a memory buffer.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param image_type Image type of the video.
   * @param dtype Data type of the video.
   * @param build_index If set to false index will not be build and some features are unavailable.
   * @param init_codecs If set to false CPU codec part is not initalized, only parser
   * @param num_frames If set, number of frames in the video.
   * @param source_info Source information for the video file.
   */
  explicit FramesDecoderBase(const char *memory_file, size_t memory_file_size,
                             DALIImageType image_type = DALI_RGB, DALIDataType dtype = DALI_UINT8,
                             bool build_index = true, bool init_codecs = true, int num_frames = -1,
                             std::string_view source_info = {});

  /**
   * @brief Number of frames in the video. It returns 0, if this information is unavailable.
   */
  int64_t NumFrames() const;

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
  virtual bool ReadNextFrame(uint8_t *data);

  /**
   * @brief Seeks to the frame given by id. Next call to ReadNextFrame will return this frame
   *
   * @param frame_id Id of the frame to seek to
   */
  virtual void SeekFrame(int frame_id);

  /**
   * @brief Decodes a collection of frames, not necessarily in ascending order, applying a boundary type
   * (what to do when sampling out of bounds).
   *
   * ISOLATED: sampling out of bounds will result in a failure.
   * CLAMP: sampling out of bounds will result in the last frame being repeated.
   * CONSTANT: sampling out of bounds will result in a constant frame being repeated (constant_frame must be provided).
   * REFLECT_1001: sampling out of bounds will result in 1001 reflection.
   * REFLECT_101: sampling out of bounds will result in 101 reflection.
   *
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
    span<float> out_timestamps = {});

  /**
   * @brief Decodes a range of evenly spaced frames, applying a boundary type
   * (what to do when sampling out of bounds).
   *
   * ISOLATED: sampling out of bounds will result in a failure.
   * CLAMP: sampling out of bounds will result in the last frame being repeated.
   * CONSTANT: sampling out of bounds will result in a constant frame being repeated (constant_frame must be provided).
   * REFLECT_1001: sampling out of bounds will result in 1001 reflection.
   * REFLECT_101: sampling out of bounds will result in 101 reflection.
   *
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
    span<float> out_timestamps = {});

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
  virtual int GetFrameIdxByTimestamp(int64_t timestamp, bool inclusive = false);

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
  bool HasIndex() const { return !index_.empty(); }

  virtual ~FramesDecoderBase() = default;
  FramesDecoderBase(FramesDecoderBase&&) = default;
  FramesDecoderBase& operator=(FramesDecoderBase&&) = default;

  std::string Filename() {
    return filename_.size() ? filename_ : "memory file";
  }

  bool IsValid() {
    return is_valid_;
  }

  const IndexEntry& Index(int frame_id) const {
    assert(HasIndex());
    return index_[frame_id];
  }

 protected:
  AVUniquePtr<AVFormatContext> ctx_;
  AVUniquePtr<AVCodecContext> codec_ctx_;
  AVUniquePtr<AVFrame> frame_;
  AVUniquePtr<AVPacket> packet_;
  std::unique_ptr<SwsContext, decltype(&sws_freeContext)> sws_ctx_{
    nullptr, sws_freeContext};

  const AVCodec *codec_ = nullptr;
  AVCodecParameters *codec_params_ = nullptr;
  int stream_id_ = -1;
  bool is_file_open_ = false;

  int OpenFile(const std::string& filename);
  int OpenMemoryFile(MemoryVideoFile& memory_video_file);

  std::vector<IndexEntry> index_;

  int next_frame_idx_ = 0;

  DALIImageType image_type_ = DALI_RGB;
  DALIDataType dtype_ = DALI_UINT8;
  VideoColorSpaceConversionType conversion_type_ = VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB;

  // False when the file doesn't have any correct content or doesn't have a valid video stream
  bool is_valid_ = false;

 private:
   /**
   * @brief Gets the packet from the decoder and reads a frame from it to provided buffer. Returns
   * boolean indicating, if the frame was succesfully read.
   * After this method returns false, there might be more frames to read. Call `ReadFlushFrame` until
   * it returns false, to get all of the frames from the video file.
   *
   * @param data Output buffer to copy data to. If nullptr, the frame will be effectively skipped.
   *
   * @returns True, if the read was succesful, or false, when all regular frames were consumed.
   *
   */
  bool ReadRegularFrame(uint8_t *data);

  /**
   * @brief Reads frames from the last packet. This packet can hold
   * multiple frames. This method will read all of them one by one.
   *
   * @param data Output buffer to copy data to. If nullptr, the frame will be effectively skipped.
   *
   * @returns True, if the read was succesful, or false, when ther are no more frames in last the packet.
   */
  bool ReadFlushFrame(uint8_t *data);

  void CopyToOutput(uint8_t *data);

  void BuildIndex();

  void DetectVariableFrameRate();

  void InitAvCodecContext();

  bool OpenAvCodec();

  bool FindVideoStream(bool require_available_avcodec = true);

  bool SelectVideoStream(int stream_id, bool require_available_avcodec);

  bool CheckDimensions();

  void ParseNumFrames();

  bool IsFormatSeekable();

  void CountFrames();

  std::string GetAllStreamInfo() const;


  int channels_ = 3;
  bool flush_state_ = false;
  bool is_vfr_ = false;

  std::string filename_ = {};
  std::unique_ptr<MemoryVideoFile> memory_video_file_;

  int num_frames_ = -1;
  bool can_seek_ = true;  // at first, we assume that the video is seekable
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_
