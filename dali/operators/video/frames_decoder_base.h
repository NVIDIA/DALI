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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "dali/core/boundary.h"
#include "dali/core/common.h"
#include "dali/core/span.h"
#include "dali/core/stream.h"
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
  // Position of the frame's packet in decoding order, counting all the packets of the stream,
  // also those decoded but not shown (e.g. hidden by an edit list); -1 if unknown
  int decode_idx = -1;
  // For a frame that SeekFrame seeks to (the last keyframe of some frame): the decoding position
  // of the packet where a seek to its timestamp lands (see AvSeekFrame); -1 otherwise or if unknown
  int seek_decode_idx = -1;
  // For a keyframe: how many leading pictures it has (shown before it, decoded after it), which
  // with FrameIndex::decode_order_timestamps are the frames right after it; -1 if unknown
  int leading_pictures = -1;
};

struct FrameIndex {
  std::string filename;
  std::vector<IndexEntry> index;
  AVRational timebase;
  /**
   * @brief The timestamps of the stream follow the order of its packets, not the order in which
   * its pictures are shown (see FramesDecoderBase::HasDecodeOrderTimestamps).
   *
   * Frame i is then the picture shown i-th, which the decoder finds by the parser's display order,
   * and the frames that a keyframe's leading pictures (shown before it, decoded after it) take are
   * not known (see FramesDecoderBase::SeekTarget).
   */
  bool decode_order_timestamps = false;

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
   * @brief Position of frame i's packet in decoding order.
   *
   * Without decoding order (e.g. a hand-built index), the frames are assumed in decoding order.
   */
  int DecodeIdx(int i) const {
    return index[i].decode_idx >= 0 ? index[i].decode_idx : i;
  }

  /**
   * @brief Decoding position where a seek to frame i starts, for a frame that SeekFrame seeks to.
   *
   * If unknown, the seek is assumed to start at the frame itself.
   */
  int SeekDecodeIdx(int i) const {
    return index[i].seek_decode_idx >= 0 ? index[i].seek_decode_idx : DecodeIdx(i);
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
 * @brief Object representing a video file. Allows access to frames and seeking.
 */
class DLL_PUBLIC FramesDecoderBase {
 public:
  /**
   * @brief Initialize the decoder from a file.
   *
   * Local paths (with no scheme or with the ``file://`` scheme) are opened directly by
   * libavformat. URLs with any other scheme (e.g. ``s3://``, ``gs://``) are opened with
   * FileStream::Open and read through the InputStream interface.
   *
   * @param filename Path or URL of a video file.
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
   * @brief Initialize the decoder from an input stream.
   *
   * @param stream Stream with video file data; the decoder takes ownership of it.
   *               The stream must cover the entire video file, including the header.
   * @param source_info Source information for the video file.
   */
  explicit FramesDecoderBase(std::unique_ptr<InputStream> stream,
                             std::string_view source_info = {},
                             DALIImageType image_type = DALI_RGB);

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
   *
   * The size from the container, unless a decoder overrides it with SetDisplaySize. Only
   * FramesDecoderGpu does: FramesIndexer and the CPU decoder keep the container size.
   */
  int Width() const {
    return display_width_ > 0 ? display_width_ : codec_params_->width;
  }

  /**
   * @brief Height of a video frame in pixels
   */
  int Height() const {
    return display_height_ > 0 ? display_height_ : codec_params_->height;
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
   * @brief Seeking forward by at most this many frames decodes through them instead of seeking.
   *
   * With an index, a decoder also decodes through to any later frame that has the same last
   * keyframe as its next frame (see SeekFrame).
   */
  static constexpr int kMinimumSeekLeap = 10;

  /**
   * @brief The most frames that the leading pictures of a keyframe can take, in an index with
   *        FrameIndex::decode_order_timestamps that does not know them (see
   *        IndexEntry::leading_pictures).
   *
   * Leading pictures are decoded after the keyframe but shown before it, so they wait for it in
   * the decoded picture buffer, which holds at most 16 pictures in H.264 and HEVC.
   */
  static constexpr int kMaxLeadingPictures = 16;

  /**
   * @brief The keyframe that SeekFrame seeks to, to return `frame` from a decoder (with an index)
   * whose next frame is `next_frame`, or -1 if it decodes through to it.
   *
   * It decodes through when `frame` is ahead of `next_frame`, by at most kMinimumSeekLeap or with
   * a last keyframe not before the seek target. Otherwise, also for any `frame` before
   * `next_frame`, it seeks to the last keyframe of `frame`. SeekFrame and
   * AddDecodeWork both take this decision from here, so the estimate cannot drift from it.
   *
   * With FrameIndex::decode_order_timestamps, the frames right after a keyframe may be the
   * pictures that lead it: shown before it, but decoded after it from pictures before it (as many
   * as IndexEntry::leading_pictures, or fewer than kMaxLeadingPictures if unknown). A decoder that
   * starts at the keyframe does not decode those, so it seeks to the keyframe before.
   *
   * @param next_frame The decoder's next frame, in range: past the end, SeekFrame resets first.
   */
  static int SeekTarget(const FrameIndex &index, int next_frame, int frame);

  /**
   * @brief Where a decoder is, for estimating its work: the frame it returns next (as
   * NextFrameIdx, -1 after the last one) and the decoding position of the last picture it
   * decoded (-1: none).
   */
  struct DecodePosition {
    int next_frame = 0;
    int decoded = -1;
  };

  /**
   * @brief Estimated work of a decoder: the pictures it goes through (those it
   * decodes and those it leaves out) and the times it reopens the file.
   *
   * A picture is a packet of the stream (see IndexEntry::decode_idx). The GPU decoder goes through
   * more than one picture for a packet that holds several, e.g. a VP9 superframe with a hidden
   * frame, or the two fields of an interlaced frame (see DecodedPictures).
   */
  struct DecodeWork {
    int64_t pictures = 0;
    int64_t reopens = 0;
  };

  /**
   * @brief The position of a decoder that read the frames up to `next_frame - 1` in a row.
   *
   * It has decoded the pictures of the frames read, and nothing after them. `next_frame` out of
   * range is the position after the last frame.
   *
   * For a decoder that seeked to reach `next_frame - 1`, this is an approximation: it also counts
   * the leading pictures of an open GOP before the seek target (at most those of one GOP), which
   * such a decoder may not have decoded.
   */
  static DecodePosition PositionAt(const FrameIndex &index, int next_frame);

  /**
   * @brief Adds the work of returning `frame` from `pos` and advances `pos`.
   *
   * Follows SeekFrame (through SeekTarget) and ReadNextFrame, counting the pictures gone through
   * (decoded_pictures_ + not_decoded_pictures_), also the hidden ones (see IndexEntry::decode_idx),
   * and a Reset as a reopening. A `frame` out of range (e.g. -1, a constant frame from HandleBoundary) is not
   * decoded: it adds nothing and leaves `pos` as it is. Not modelled: a failed av_seek_frame
   * (after which every seek resets the decoder), and the RASL pictures of an HEVC CRA, which are
   * counted after a seek to it although a decoder skips them there (they reference pictures
   * before the seek).
   *
   * The CPU decoder accepts only codecs without frame reordering (VP8, VP9, MJPEG, see
   * FramesDecoderCpu::SelectVideoStream), where the estimate is exact. If it ever accepted a
   * codec with reordering (e.g. H.264), libavcodec could need packets beyond a frame's own
   * position before it returns the frame (its decoder delay), and the CPU would decode more than
   * estimated.
   */
  static void AddDecodeWork(const FrameIndex &index, DecodePosition &pos, int frame,
                            DecodeWork &work);

  /**
   * @brief Handles boundary conditions for a given frame index.
   *
   * @param boundary_type The boundary type to apply.
   * @param frame_id The frame index to handle.
   * @param roi_start The start of the region of interest.
   * @param roi_end The end of the region of interest.
   */
  static int HandleBoundary(boundary::BoundaryType boundary_type, int frame_id,
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
    return index_.GetFrameIdxByTimestamp(timestamp, inclusive);
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
  bool HasIndex() const { return index_.size() > 0; }

  /**
   * @brief Builds the index of the video file.
   */
  virtual void BuildIndex();

  const FrameIndex& GetIndex() const {
    return index_;
  }

  void SetIndex(const FrameIndex& index) {
    // sorted by timestamp, as BuildIndex makes it (the GPU decoder searches it). Checked in release
    // builds too: an unsorted index makes the decoder drop frames it should read.
    if (!std::is_sorted(index.index.begin(), index.index.end(),
                        [](const IndexEntry &a, const IndexEntry &b) { return a.pts < b.pts; })) {
      throw std::invalid_argument("The frame index must be sorted by timestamp.");
    }
    index_ = index;
    num_frames_ = index.size();
  }

  virtual ~FramesDecoderBase() = default;
  FramesDecoderBase(FramesDecoderBase&&) = default;
  FramesDecoderBase& operator=(FramesDecoderBase&&) = default;

  std::string Filename() {
    return filename_.size() ? filename_ : input_stream_ ? "input stream" : "memory file";
  }

  bool IsValid() {
    return is_valid_;
  }

  void SetOutputType(DALIDataType dtype) {
    dtype_ = dtype;
  }

  /**
   * @brief Number of pictures decoded so far (for tests and diagnostics).
   *
   * The CPU decoder counts the packets it sends to FFmpeg, the GPU decoder the pictures that
   * NVDEC decodes: more than the packets when a packet holds several pictures (see DecodeWork).
   * The pictures that the GPU decoder does not decode at all are not counted (see
   * not_decoded_pictures_); DecodeWork estimates both.
   */
  int64_t DecodedPictures() const {
    return decoded_pictures_;
  }

  /**
   * @brief Number of times the file was opened again by Reset (for tests and diagnostics).
   *
   * Includes the resets done after counting frames or building the index.
   */
  int64_t Reopens() const {
    return reopens_;
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

  /**
   * @brief Whether the timestamps of the selected stream follow the order of its packets, which is
   *        not the order in which its pictures are shown.
   *
   * AVI stores no presentation timestamps: the demuxer stamps each packet with its position in the
   * stream (as pts and dts for MPEG-4 Part 2, as dts only for H.264 and HEVC). With a codec that
   * reorders pictures, that is the decoding order: also for DivX packed B-frames, once the decoder
   * unpacks them (mpeg4_unpack_bframes moves each B-frame to the next packet, which keeps its
   * timestamp).
   */
  bool HasDecodeOrderTimestamps() const;

  /**
   * @brief Overrides the size reported by Width() and Height(), when the size of the pictures in
   *        the stream differs from the one declared by the container.
   */
  void SetDisplaySize(int width, int height) {
    display_width_ = width;
    display_height_ = height;
  }

  AVUniquePtr<AVFormatContext> ctx_;
  AVUniquePtr<AVCodecContext> codec_ctx_;
  AVUniquePtr<AVFrame> frame_;
  AVUniquePtr<AVPacket> packet_;
  AVCodecParameters *codec_params_ = nullptr;
  int stream_id_ = -1;
  bool is_file_open_ = false;

  int OpenFile(const std::string& filename);
  int OpenMemoryFile(MemoryVideoFile& memory_video_file);
  int OpenInputStream(InputStream& stream);

  FrameIndex index_;

  int next_frame_idx_ = 0;

  DALIImageType image_type_ = DALI_RGB;
  DALIDataType dtype_ = DALI_UINT8;

  // False when the file doesn't have any correct content or doesn't have a valid video stream
  bool is_valid_ = false;

  int64_t decoded_pictures_ = 0;  // see DecodedPictures
  int64_t reopens_ = 0;  // see Reopens
  // The pictures that the GPU decoder goes through without decoding: H.264 non-reference pictures
  // that are never read with their pixels (see FramesDecoderGpu::ShouldDecodePicture). The CPU
  // decoder decodes every picture: 0. Together with decoded_pictures_, the pictures that DecodeWork
  // estimates. Read by tests through FramesDecoderTestAccess.
  int64_t not_decoded_pictures_ = 0;

  /**
   * @brief The frames read with their data next, in order, when known: DecodeFrames sets those it
   * has yet to read and SeekFrame the one it seeks to. Empty: any frame may be read.
   *
   * A frame decoded on the way to them that is not one of them is only skipped.
   */
  span<const int> frames_to_read_;

  /**
   * @brief Declares the frames read with their data next (frames_to_read_) for the lifetime of the
   *        object, and clears the declaration when the outermost scope ends.
   *
   * A scope created while another one is alive (SeekFrame called by DecodeFrames) leaves the
   * declaration of the outer one alone. The frames passed to Declare must outlive the scope.
   */
  class FramesToReadScope {
   public:
    explicit FramesToReadScope(FramesDecoderBase &decoder)
    : decoder_(decoder), outermost_(decoder.frames_to_read_.empty()) {}

    ~FramesToReadScope() {
      if (outermost_)
        decoder_.frames_to_read_ = {};
    }

    FramesToReadScope(const FramesToReadScope &) = delete;
    FramesToReadScope &operator=(const FramesToReadScope &) = delete;

    void Declare(span<const int> frames) {
      if (outermost_)
        decoder_.frames_to_read_ = frames;
    }

   private:
    FramesDecoderBase &decoder_;
    bool outermost_;
  };

 private:
  friend struct FramesDecoderTestAccess;  // reads the counters of the pictures (in tests)

  void DetectVariableFrameRate();

  bool CheckDimensions();

  void ParseNumFrames();

  /**
   * @brief Returns the error that made av_read_frame return `ret`, or 0 at the end of the file.
   */
  int ReadFrameError(int ret) const;

  bool IsFormatSeekable();

  std::string GetAllStreamInfo() const;

  /**
   * @brief Opens the format context with custom I/O, reading through the provided callbacks.
   */
  int OpenCustomIO(void *opaque,
                   int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                   int64_t (*seek)(void *opaque, int64_t offset, int whence),
                   int buffer_size);

  int channels_ = 3;
  int display_width_ = 0;
  int display_height_ = 0;
  bool is_vfr_ = false;

  std::string filename_ = {};
  std::unique_ptr<MemoryVideoFile> memory_video_file_;
  std::unique_ptr<InputStream> input_stream_;

  int num_frames_ = -1;
  bool can_seek_ = true;  // at first, we assume that the video is seekable
};

/**
 * @brief Reads the frame index of a video file, without decoding it.
 *
 * It only demuxes the file: it has no codec context, parser or frame buffers, so it is much
 * cheaper to create than a decoder. It is valid for the files whose container has a video stream.
 * Whether a backend can decode the codec of that stream is up to the caller to check, for example
 * with `Decoder::IsCodecSupported(indexer.CodecId())`.
 */
class DLL_PUBLIC FramesIndexer : public FramesDecoderBase {
 public:
  explicit FramesIndexer(const std::string &filename, DALIImageType image_type = DALI_RGB);

  /**
   * @brief Codec of the selected video stream. Only valid when IsValid().
   */
  AVCodecID CodecId() const {
    return codec_params_->codec_id;
  }

  bool ReadNextFrame(uint8_t *data) override;

  void CopyFrame(uint8_t *dst, const uint8_t *src) override;

  void Flush() override {}

  /**
   * @brief Does nothing: the file is not read after it is indexed, so it is not reopened, as
   *        FramesDecoderBase::BuildIndex does at its end.
   */
  void Reset() override {}
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_BASE_H_
