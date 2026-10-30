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

#ifndef DALI_OPERATORS_VIDEO_FRAMES_DEMUXER_H_
#define DALI_OPERATORS_VIDEO_FRAMES_DEMUXER_H_

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavcodec/bsf.h>
}

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/unique_handle.h"

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
   * The entries must be sorted by pts, which BuildIndex guarantees.
   *
   * @param timestamp Timestamp of the frame to seek to
   * @param rounddown If true, the seek will be to a frame that has this timestamp or a previous one
   */
  int GetFrameIdxByTimestamp(int64_t timestamp, bool rounddown = false) const {
    LOG_LINE << "GetFrameIdxByTimestamp: timestamp=" << timestamp << ", rounddown=" << rounddown
             << ", index_size=" << index.size() << std::endl;
    // The first frame with pts >= timestamp
    auto it = std::lower_bound(index.begin(), index.end(), timestamp,
                               [](const IndexEntry &e, int64_t ts) { return e.pts < ts; });
    int frame_idx = 0;  // returned when the timestamp is past the last frame
  #if LOGS_ENABLED
    std::optional<int64_t> pts_found = {};
  #endif
    if (it != index.end()) {
      frame_idx = it - index.begin();
      if (it->pts > timestamp && rounddown && frame_idx > 0)
        frame_idx--;
    #if LOGS_ENABLED
      pts_found = index[frame_idx].pts;
    #endif
    }
    assert(frame_idx >= 0 && frame_idx < static_cast<int>(index.size()));
  #if LOGS_ENABLED
    if (pts_found) {
      LOG_LINE << "Returning frame_idx=" << frame_idx << " with pts=" << *pts_found << std::endl;
    } else {
      LOG_LINE << "Returning frame_idx=" << frame_idx << " (frame not found)" << std::endl;
    }
  #endif
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

static void DestroyAvObject(AVBSFContext **bsf) {
  assert(*bsf != nullptr);
  av_bsf_free(bsf);
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
 * @brief Reads the container of a video file: the frame index and the packets of its video stream.
 *
 * It owns the format context, the packet, the selected stream, the frame index and the number of
 * frames, and knows nothing about decoding. The stream is not selected on construction: call
 * SelectVideoStream before using anything that depends on it (dimensions, timebase, packets,
 * the number of frames, the index).
 */
class DLL_PUBLIC FramesDemuxer {
 public:
  /**
   * @brief Opens a video file.
   *
   * @param filename Path to a video file.
   */
  explicit FramesDemuxer(const std::string &filename);

  /**
   * @brief Opens a video file from a memory buffer, which has to outlive the demuxer.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param source_info Source information for the video file.
   */
  FramesDemuxer(const char *memory_file, size_t memory_file_size,
                std::string_view source_info = {});

  FramesDemuxer(FramesDemuxer&&) = default;
  FramesDemuxer& operator=(FramesDemuxer&&) = default;

  /**
   * @brief False when the file could not be opened. Whether it has a usable video stream is
   *        the result of SelectVideoStream.
   */
  bool IsValid() const {
    return is_valid_;
  }

  std::string Filename() const {
    return filename_.size() ? filename_ : "memory file";
  }

  /**
   * @brief Selects the video stream. If stream_id is -1, the video stream will be selected
   *        automatically.
   *
   * @return False if there is no usable video stream.
   */
  bool SelectVideoStream(int stream_id = -1);

  int StreamId() const {
    return stream_id_;
  }

  /**
   * @brief Parameters of the selected video stream.
   */
  const AVCodecParameters *CodecParams() const {
    return codec_params_;
  }

  const AVInputFormat *InputFormat() const {
    return ctx_->iformat;
  }

  int Width() const {
    return codec_params_->width;
  }

  int Height() const {
    return codec_params_->height;
  }

  AVRational GetTimebase() const {
    return ctx_->streams[stream_id_]->time_base;
  }

  /**
   * @brief True if the number of frames is known without parsing the file: it was set, or the
   *        container says it.
   */
  bool NumFramesKnown();

  /**
   * @brief Number of frames in the video, or -1 if it is not known yet. It becomes known after
   *        NumFramesKnown() returns true, or after ParseNumFrames(), BuildIndex() or
   *        SetNumFrames() is called.
   */
  int64_t NumFrames() const {
    return num_frames_;
  }

  /**
   * @brief Counts the packets of the video stream. The file is left at its end.
   */
  void ParseNumFrames();

  /**
   * @brief Set the number of frames in the video, which can be used to avoid parsing the file.
   */
  void SetNumFrames(int64_t num_frames) {
    num_frames_ = num_frames;
  }

  /**
   * @brief Builds the index of the video file (does nothing if there is one). The file is left at
   *        its end.
   */
  void BuildIndex();

  bool HasIndex() const {
    return index_.size() > 0;
  }

  const FrameIndex& GetIndex() const {
    return index_;
  }

  void SetIndex(const FrameIndex& index) {
    index_ = index;
    num_frames_ = index.size();
  }

  /**
   * @brief Is video variable frame rate. Known when the index was built.
   */
  bool IsVfr() const {
    return is_vfr_;
  }

  /**
   * @brief Reads the next packet of the file, of any stream, into the packet of the demuxer.
   *
   * @return 0 on success, a negative AVERROR (e.g. AVERROR_EOF at the end of the file) otherwise
   */
  int ReadFrame();

  /**
   * @brief The packet that ReadFrame reads into. The caller has to unreference it.
   */
  AVPacket *Packet() const {
    return packet_.get();
  }

  /**
   * @brief Seeks to the keyframe at or before the timestamp, using av_seek_frame
   *
   * @param timestamp Timestamp in the timebase of the stream.
   * @return Boolean indicating whether the seek was successful. Once a seek fails, the file is
   *         not seekable, until it is reopened.
   */
  bool Seek(int64_t timestamp);

  /**
   * @brief Reopens the file, so that it is read from the start. The stream has to be selected
   *        again with SelectVideoStream. Throws if the file can't be opened.
   */
  void Reopen();

 private:
  int OpenFile(const std::string& filename);
  int OpenMemoryFile(MemoryVideoFile& memory_video_file);

  void DetectVariableFrameRate();

  bool CheckDimensions();

  std::string GetAllStreamInfo() const;

  AVUniquePtr<AVFormatContext> ctx_;
  AVUniquePtr<AVPacket> packet_;
  AVCodecParameters *codec_params_ = nullptr;
  int stream_id_ = -1;

  FrameIndex index_;
  bool is_vfr_ = false;

  std::string filename_ = {};
  std::unique_ptr<MemoryVideoFile> memory_video_file_;

  int num_frames_ = -1;
  bool can_seek_ = true;  // at first, we assume that the video is seekable

  // False when the file couldn't be opened
  bool is_valid_ = false;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DEMUXER_H_
