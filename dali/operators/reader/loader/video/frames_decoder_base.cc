// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/video/frames_decoder_base.h"
#include <iomanip>
#include <memory>
#include "dali/core/error_handling.h"
#include "dali/core/util.h"

namespace dali {

int MemoryVideoFile::Read(unsigned char *buffer, int buffer_size) {
  int left_in_file = size_ - position_;
  if (left_in_file <= 0) {
    return AVERROR_EOF;
  }

  int to_read = std::min(left_in_file, buffer_size);
  std::copy(data_ + position_, data_ + position_ + to_read, buffer);
  position_ += to_read;
  return to_read;
}

/**
 * @brief Method for seeking the memory video. It sets position according to provided arguments.
 *
 * @param new_position Requested new_position.
 * @param mode Chosen method of seeking. This argument changes how new_position is interpreted and
 * how seeking is performed.
 * @return int64_t actual new position in the file.
 */
int64_t MemoryVideoFile::Seek(int64_t new_position, int mode) {
  switch (mode) {
    case SEEK_SET:
      position_ = new_position;
      break;
    case AVSEEK_SIZE:
      return size_;

    default:
      DALI_FAIL(make_string(
          "Unsupported seeking method in FramesDecoderBase from memory file. Seeking method: ",
          mode));
  }

  return position_;
}

namespace detail {
std::string av_error_string(int ret) {
  static char msg[AV_ERROR_MAX_STRING_SIZE];
  memset(msg, 0, sizeof(msg));
  return std::string(av_make_error_string(msg, AV_ERROR_MAX_STRING_SIZE, ret));
}

int read_memory_video_file(void *data_ptr, uint8_t *av_io_buffer, int av_io_buffer_size) {
  MemoryVideoFile *memory_video_file = static_cast<MemoryVideoFile *>(data_ptr);
  return memory_video_file->Read(av_io_buffer, av_io_buffer_size);
}

int64_t seek_memory_video_file(void *data_ptr, int64_t new_position, int origin) {
  MemoryVideoFile *memory_video_file = static_cast<MemoryVideoFile *>(data_ptr);
  return memory_video_file->Seek(new_position, origin);
}

}  // namespace detail

using AVPacketScope = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;

int64_t FramesDecoderBase::NumFrames() const {
  if (num_frames_.has_value()) {
    return num_frames_.value();
  }

  if (index_.has_value()) {
    return index_->size();
  }

  return av_state_->ctx_->streams[av_state_->stream_id_]->nb_frames;
}

void FramesDecoderBase::InitAvState(bool init_codecs) {
  av_state_->codec_ctx_ = avcodec_alloc_context3(av_state_->codec_);
  DALI_ENFORCE(av_state_->codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(av_state_->codec_ctx_, av_state_->codec_params_);
  DALI_ENFORCE(ret >= 0, make_string("Could not fill the codec based on parameters: ",
                                     detail::av_error_string(ret)));

  av_state_->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state_->packet_, "Could not allocate av packet");

  if (init_codecs) {
    ret = avcodec_open2(av_state_->codec_ctx_, av_state_->codec_, nullptr);
    DALI_ENFORCE(ret == 0,
                 make_string("Could not initialize codec context: ", detail::av_error_string(ret)));

    av_state_->frame_ = av_frame_alloc();
    DALI_ENFORCE(av_state_->frame_, "Could not allocate the av frame");
  }
}

bool FramesDecoderBase::FindVideoStream(bool init_codecs) {
  if (init_codecs) {
    size_t i = 0;

    for (i = 0; i < av_state_->ctx_->nb_streams; ++i) {
      av_state_->codec_params_ = av_state_->ctx_->streams[i]->codecpar;
      av_state_->codec_ = avcodec_find_decoder(av_state_->codec_params_->codec_id);
      if (av_state_->codec_ == nullptr) {
        continue;
      }
      if (av_state_->codec_->type == AVMEDIA_TYPE_VIDEO) {
        LOG_LINE << "Found video stream " << i << " with codec " << av_state_->codec_->name
                 << std::endl;
        av_state_->stream_id_ = i;
        break;
      }
    }

    if (i >= av_state_->ctx_->nb_streams) {
      DALI_WARN(make_string("Could not find a valid video stream in a file ", Filename()));
      return false;
    }
  } else {
    av_state_->stream_id_ = av_find_best_stream(av_state_->ctx_, AVMEDIA_TYPE_VIDEO,
                                                -1, -1, nullptr, 0);

    LOG_LINE << "Best stream " << av_state_->stream_id_ << std::endl;
    if (av_state_->stream_id_ < 0) {
      DALI_WARN(make_string("Could not find a valid video stream in a file ", Filename()));
      return false;
    }

    av_state_->codec_params_ = av_state_->ctx_->streams[av_state_->stream_id_]->codecpar;
  }
  if (Height() == 0 || Width() == 0) {
    if (avformat_find_stream_info(av_state_->ctx_, nullptr) < 0) {
      DALI_WARN(make_string("Could not find stream information in ", Filename()));
      return false;
    }
    if (Height() == 0 || Width() == 0) {
      DALI_WARN("Couldn't load video size info.");
      return false;
    }
  }
  return true;
}

FramesDecoderBase::FramesDecoderBase(const std::string &filename)
    : av_state_(std::make_unique<AvState>()) {
  av_log_set_level(AV_LOG_ERROR);
  filename_ = filename;
  av_state_->ctx_ = avformat_alloc_context();
  DALI_ENFORCE(av_state_->ctx_, "Could not alloc avformat context");

  int ret = avformat_open_input(&av_state_->ctx_, Filename().c_str(), nullptr, nullptr);
  if (ret != 0) {
    DALI_WARN(make_string("Failed to open video file \"", Filename(), "\" due to ",
                          detail::av_error_string(ret)));
    return;
  }

  if (!FindVideoStream()) {
    return;
  }

  InitAvState();
  BuildIndex();
  is_valid_ = true;
}

FramesDecoderBase::FramesDecoderBase(const char *memory_file, int memory_file_size,
                                     bool build_index, bool init_codecs, int num_frames,
                                     std::string_view source_info)
    : av_state_(std::make_unique<AvState>()) {
  av_log_set_level(AV_LOG_ERROR);
  filename_ = source_info;
  memory_video_file_.emplace(memory_file, memory_file_size);

  DALI_ENFORCE(init_codecs || !build_index,
               "FramesDecoderBase doesn't support index without CPU codecs");
  av_log_set_level(AV_LOG_ERROR);

  if (num_frames != -1) {
    num_frames_ = num_frames;
  }

  av_state_->ctx_ = avformat_alloc_context();
  DALI_ENFORCE(av_state_->ctx_, "Could not alloc avformat context");

  uint8_t *av_io_buffer = static_cast<uint8_t *>(av_malloc(default_av_buffer_size));

  AVIOContext *av_io_context = avio_alloc_context(
    av_io_buffer,
    default_av_buffer_size,
    0,
    &memory_video_file_.value(),
    detail::read_memory_video_file,
    nullptr,
    detail::seek_memory_video_file);

  av_state_->ctx_->pb = av_io_context;

  int ret = avformat_open_input(&av_state_->ctx_, "", nullptr, nullptr);
  if (ret != 0) {
    // av_state_->ctx_ is nullified so we need to free the memory here instead of the
    // AvState destructor which cannot access it through av_state_->ctx_ anymore
    av_freep(&av_io_context->buffer);
    avio_context_free(&av_io_context);
    DALI_WARN(make_string("Failed to open video file \"", Filename(), "\", due to ",
                          detail::av_error_string(ret)));
    return;
  }

  if (!FindVideoStream(init_codecs || build_index)) {
    return;
  }

  InitAvState(init_codecs || build_index);

  // Number of frames is unknown and we do not plan to build the index
  if (NumFrames() == 0 && !build_index) {
    ParseNumFrames();
  }

  if (build_index) {
    LOG_LINE << "Building index" << std::endl;
    BuildIndex();
  }
  is_valid_ = true;
}

void FramesDecoderBase::CreateAvState(std::unique_ptr<AvState> &av_state, bool init_codecs) {
  av_state->ctx_ = avformat_alloc_context();
  DALI_ENFORCE(av_state_->ctx_, "Could not alloc avformat context");

  uint8_t *av_io_buffer = static_cast<uint8_t *>(av_malloc(default_av_buffer_size));

  AVIOContext *av_io_context = avio_alloc_context(
      av_io_buffer,
      default_av_buffer_size,
      0,
      &memory_video_file_.value(),
      detail::read_memory_video_file,
      nullptr,
      detail::seek_memory_video_file);

  av_state->ctx_->pb = av_io_context;

  int ret = avformat_open_input(&av_state->ctx_, "", nullptr, nullptr);
  if (ret != 0) {
    // av_state_->ctx_ is nullified so we need to free the memory here instead of the
    // AvState destructor which cannot access it through av_state_->ctx_ anymore
    av_freep(&av_io_context->buffer);
    avio_context_free(&av_io_context);
    DALI_FAIL(make_string("Failed to open video file \"", Filename(), "\", due to ",
                          detail::av_error_string(ret)));
  }
  av_state->stream_id_ =
      av_find_best_stream(av_state->ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  av_state->codec_params_ = av_state->ctx_->streams[av_state->stream_id_]->codecpar;

  av_state->codec_ctx_ = avcodec_alloc_context3(av_state->codec_);
  DALI_ENFORCE(av_state->codec_ctx_, "Could not alloc av codec context");

  ret = avcodec_parameters_to_context(av_state->codec_ctx_, av_state->codec_params_);
  DALI_ENFORCE(ret >= 0, make_string("Could not fill the codec based on parameters: ",
                                     detail::av_error_string(ret)));

  av_state->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state->packet_, "Could not allocate av packet");
}

void FramesDecoderBase::ParseNumFrames() {
  if (IsFormatSeekable()) {
    CountFrames(av_state_.get());
    Reset();
  } else {
    // Failover for unseekable video
    auto current_position = memory_video_file_->position_;
    memory_video_file_->Seek(0, SEEK_SET);
    std::unique_ptr<AvState> tmp_av_state = std::make_unique<AvState>();
    CreateAvState(tmp_av_state, false);
    CountFrames(tmp_av_state.get());
    memory_video_file_->Seek(current_position, SEEK_SET);
  }
}

void FramesDecoderBase::CountFrames(AvState *av_state) {
  num_frames_ = 0;
  while (true) {
    int ret = av_read_frame(av_state->ctx_, av_state->packet_);
    auto packet = AVPacketScope(av_state->packet_, av_packet_unref);
    if (ret != 0) {
      break;  // End of file
    }

    if (packet->stream_index != av_state->stream_id_) {
      continue;
    }
    ++num_frames_.value();
  }
}

IMPL_HAS_MEMBER(read_seek);
IMPL_HAS_MEMBER(read_seek2);

template <typename FormatDesc>
bool IsFormatSeekableHelper(FormatDesc *iformat) {
  if constexpr (has_member_read_seek_v<FormatDesc>) {
    static_assert(has_member_read_seek2_v<FormatDesc>);
    if (iformat->read_seek == nullptr && iformat->read_seek2 == nullptr) {
      return false;
    }
  } else {
    if (iformat->flags & (AVFMT_NOBINSEARCH | AVFMT_NOGENSEARCH)) {
      return false;
    }
  }
  return true;
}

bool FramesDecoderBase::IsFormatSeekable() {
  if (!IsFormatSeekableHelper(av_state_->ctx_->iformat)) {
    return false;
  }
  return av_state_->ctx_->pb->read_seek != nullptr;
}

/**
 * @brief Reads the length of a Network Abstraction Layer (NAL) unit from a buffer.
 *
 * NAL units are the basic elements of H.264/AVC and H.265/HEVC video compression standards.
 * In the Annex B byte stream format, NAL units are prefixed with a 4-byte length field
 * that indicates the size of the following NAL unit.
 *
 * This function reads those 4 bytes in big-endian order to get the NAL unit length.
 *
 * Reference: ITU-T H.264 and H.265 specifications
 *
 * @param buf Pointer to the buffer containing the NAL unit length prefix
 * @return The length of the NAL unit in bytes
 */
static inline uint32_t read_nal_unit_length(uint8_t *buf) {
  uint32_t length = 0;
  length = (buf)[0] << 24 | (buf)[1] << 16 | (buf)[2] << 8 | (buf)[3];
  return length;
}

void FramesDecoderBase::BuildIndex() {
  LOG_LINE << "Starting BuildIndex()" << std::endl;

  // Initialize empty index vector to store frame metadata
  index_ = std::vector<IndexEntry>();

  // Track the position of the last keyframe seen
  int last_keyframe = -1;
  int frame_count = 0;

  while (true) {
    // Read the next frame from the video
    int ret = av_read_frame(av_state_->ctx_, av_state_->packet_);
    auto packet = AVPacketScope(av_state_->packet_, av_packet_unref);

    if (ret != 0) {
      LOG_LINE << "End of file reached after " << frame_count << " frames" << std::endl;
      break;  // Just break when we hit EOF instead of trying to seek back
    }

    frame_count++;

    // Skip packets from other streams (e.g. audio)
    if (packet->stream_index != av_state_->stream_id_) {
      continue;
    }

    IndexEntry entry;
    entry.is_keyframe = false;  // Default to false, set true only if confirmed

    // Check if this packet contains a keyframe
    if (packet->flags & AV_PKT_FLAG_KEY) {
      LOG_LINE << "Found potential keyframe at frame " << frame_count << std::endl;

      // Special handling for H.264 and HEVC formats
      auto codec_id = av_state_->ctx_->streams[packet->stream_index]->codecpar->codec_id;
      if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC) {
        // Parse NAL units to verify if this is actually a keyframe
        // NAL = Network Abstraction Layer, the basic unit of encoded video
        const uint8_t *end = packet->data + packet->size;
        uint8_t *nal_start = packet->data;

        // Iterate through NAL units in the packet
        while (nal_start + 4 < end) {
          // Each NAL unit is prefixed with a 4-byte length
          uint32_t nal_size = read_nal_unit_length(nal_start);
          nal_start += 4;
          if (nal_start + nal_size > end) {
            break;
          }

          if (codec_id == AV_CODEC_ID_H264) {
            // In H.264, the NAL unit type is in the lower 5 bits
            uint8_t nal_unit_type = nal_start[0] & 0x1F;
            // Type 5 indicates an IDR frame (Instantaneous Decoding Refresh)
            // IDR frames are special keyframes that clear all reference buffers
            if (nal_unit_type == 5) {
              entry.is_keyframe = true;
              break;
            }
          } else {  // AV_CODEC_ID_HEVC
            // In HEVC/H.265, NAL unit type is in bits 1-6 of the first byte
            uint8_t nal_unit_type = (nal_start[0] >> 1) & 0x3F;
            // Types 16-21 are IRAP (Intra Random Access Point) pictures
            // which serve as keyframes in HEVC
            if (nal_unit_type >= 16 && nal_unit_type <= 21) {
              entry.is_keyframe = true;
              break;
            }
          }
          nal_start += nal_size;  // Advance to next NAL unit
        }
      } else {
        // For other codecs, trust the AV_PKT_FLAG_KEY flag
        entry.is_keyframe = true;
      }
    }

    // Store presentation timestamp (pts) or decode timestamp (dts) if pts not available
    entry.pts = (packet->pts != AV_NOPTS_VALUE) ? packet->pts : packet->dts;
    if (entry.pts == AV_NOPTS_VALUE) {
      DALI_FAIL(make_string("Video file \"", Filename(), "\" has no valid timestamps"));
    }

    // Update last keyframe position if this is a keyframe
    if (entry.is_keyframe) {
      last_keyframe = index_->size();
    }
    entry.last_keyframe_id = last_keyframe;

    // Regular frame, not a flush frame
    entry.is_flush_frame = false;
    index_->push_back(entry);
  }

  LOG_LINE << "Index building complete. Total frames: " << index_->size() << std::endl;

  DALI_ENFORCE(!index_->empty(),
               make_string("No valid frames found in video file \"", Filename(), "\""));

  // Mark last frame as flush frame
  index_->back().is_flush_frame = true;

  // Sort frames by presentation timestamp
  // This is needed because frames may be stored out of order in the container
  std::sort(index_->begin(), index_->end(),
            [](const IndexEntry &a, const IndexEntry &b) { return a.pts < b.pts; });

  // After sorting, we need to update last_keyframe_id references
  std::vector<int> keyframe_positions;
  for (size_t i = 0; i < index_->size(); i++) {
    if ((*index_)[i].is_keyframe) {
      keyframe_positions.push_back(i);
    }
  }

  DALI_ENFORCE(!keyframe_positions.empty(),
               make_string("No keyframes found in video file \"", Filename(), "\""));

  // Update last_keyframe_id for each frame after sorting
  for (size_t i = 0; i < index_->size(); i++) {
    // Find the last keyframe that comes before or at this frame
    auto it = std::upper_bound(keyframe_positions.begin(), keyframe_positions.end(), i);
    if (it == keyframe_positions.begin()) {
      (*index_)[i].last_keyframe_id = 0;  // First keyframe
    } else {
      (*index_)[i].last_keyframe_id = *(--it);
    }
  }

  // Detect if video has variable frame rate (VFR)
  DetectVariableFrameRate();
  Reset();
}

void FramesDecoderBase::DetectVariableFrameRate() {
  is_vfr_ = false;
  if (index_->size() > 3) {
    int64_t pts_step = (*index_)[1].pts - (*index_)[0].pts;
    for (size_t i = 2; i < index_->size(); i++) {
      if (((*index_)[i].pts - (*index_)[i-1].pts) != pts_step) {
        is_vfr_ = true;
        break;
      }
    }
  }
}

void FramesDecoderBase::CopyToOutput(uint8_t *data) {
  LazyInitSwContext();

  uint8_t *dest[4] = {data, nullptr, nullptr, nullptr};
  int dest_linesize[4] = {av_state_->frame_->width * Channels(), 0, 0, 0};

  int ret = sws_scale(
    av_state_->sws_ctx_,
    av_state_->frame_->data,
    av_state_->frame_->linesize,
    0,
    av_state_->frame_->height,
    dest,
    dest_linesize);

  DALI_ENFORCE(ret >= 0,
               make_string("Could not convert frame data to RGB: ", detail::av_error_string(ret)));
}

void FramesDecoderBase::LazyInitSwContext() {
  if (av_state_->sws_ctx_ == nullptr) {
    av_state_->sws_ctx_ = sws_getContext(
      Width(),
      Height(),
      av_state_->codec_ctx_->pix_fmt,
      Width(),
      Height(),
      AV_PIX_FMT_RGB24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr);
    DALI_ENFORCE(av_state_->sws_ctx_, "Could not create sw context");
  }
}

bool FramesDecoderBase::ReadRegularFrame(uint8_t *data, bool copy_to_output) {
  int ret = -1;
  while (true) {
    ret = av_read_frame(av_state_->ctx_, av_state_->packet_);
    auto packet = AVPacketScope(av_state_->packet_, av_packet_unref);
    if (ret != 0) {
      break;  // End of file
    }
    if (packet->stream_index != av_state_->stream_id_) {
      continue;
    }
    ret = avcodec_send_packet(av_state_->codec_ctx_, packet.get());
    DALI_ENFORCE(ret >= 0,
                 make_string("Failed to send packet to decoder: ", detail::av_error_string(ret)));

    ret = avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_);
    if (ret == AVERROR(EAGAIN)) {
      continue;
    }

    if (ret == AVERROR_EOF) {
      break;
    }

    LOG_LINE << "Read frame (ReadRegularFrame), index " << next_frame_idx_ << ", timestamp "
             << std::setw(5) << av_state_->frame_->pts << ", current copy " << copy_to_output
             << std::endl;
    if (!copy_to_output) {
      ++next_frame_idx_;
      return true;
    }

    CopyToOutput(data);
    ++next_frame_idx_;
    return true;
  }

  ret = avcodec_send_packet(av_state_->codec_ctx_, nullptr);
  DALI_ENFORCE(ret >= 0,
               make_string("Failed to send packet to decoder: ", detail::av_error_string(ret)));
  flush_state_ = true;

  return false;
}

void FramesDecoderBase::Reset() {
  LOG_LINE << "Resetting decoder" << std::endl;

  next_frame_idx_ = 0;

  if (flush_state_) {
    flush_state_ = false;
  }

  int ret = av_seek_frame(av_state_->ctx_, av_state_->stream_id_, 0, AVSEEK_FLAG_FRAME);
  DALI_ENFORCE(ret >= 0, make_string("Could not seek to the first frame of video \"", Filename(),
                                     "\", due to ", detail::av_error_string(ret)));
  if (av_state_->codec_) {
    avcodec_flush_buffers(av_state_->codec_ctx_);
  }
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

  // If we are seeking to a frame that is before the current frame,
  // or we are seeking to a frame that is more than 10 frames away,
  // or the current frame index is invalid (e.g. end of file),
  // we will to seek to the nearest keyframe first
  LOG_LINE << "SeekFrame: frame_id=" << frame_id << ", next_frame_idx_=" << next_frame_idx_
           << std::endl;
  if (frame_id < next_frame_idx_ || frame_id > next_frame_idx_ + 10 || next_frame_idx_ < 0) {
    // If we have an index, we can seek to the nearest keyframe first
    if (HasIndex()) {
      LOG_LINE << "Using index to find nearest keyframe" << std::endl;
      const auto &current_frame = Index(next_frame_idx_);
      const auto &requested_frame = Index(frame_id);
      auto keyframe_id = requested_frame.last_keyframe_id;
      // if we are seeking to a different keyframe than the current frame,
      // or if we are seeking to a frame that is before the current frame,
      // we need to seek to the keyframe first
      if (current_frame.last_keyframe_id != keyframe_id || frame_id < next_frame_idx_) {
        // We are seeking to a different keyframe than the current frame,
        // so we need to seek to the keyframe first
        auto &keyframe_entry = Index(keyframe_id);
        LOG_LINE << "Seeking to key frame " << keyframe_id << " timestamp " << keyframe_entry.pts
                 << " for requested frame " << frame_id << " timestamp " << requested_frame.pts
                 << std::endl;

        // Seeking clears av buffers, so reset flush state info
        if (flush_state_) {
          while (ReadFlushFrame(nullptr, false)) {}
          flush_state_ = false;
        }

        int ret = av_seek_frame(av_state_->ctx_, av_state_->stream_id_, keyframe_entry.pts,
                                AVSEEK_FLAG_FRAME);
        DALI_ENFORCE(ret >= 0, make_string("Failed to seek to keyframe", keyframe_id, "in video \"",
                                           Filename(), "\" due to ", detail::av_error_string(ret)));
        avcodec_flush_buffers(av_state_->codec_ctx_);
        next_frame_idx_ = keyframe_id;
      }
    } else if (frame_id < next_frame_idx_) {
      LOG_LINE << "No index available and seeking backwards, resetting decoder" << std::endl;
      // If we are seeking to a frame that is before the current frame and there's no index,
      // we need to reset the decoder
      if (IsFormatSeekable()) {
        Reset();
      } else {
        DALI_FAIL(make_string("Video file \"", Filename(), "\" is not seekable"));
        // TODO(janton): Implement seeking by closing and reopening the handle
      }
    }
  }
  assert(next_frame_idx_ <= frame_id);
  // Skip all remaining frames until the requested frame
  LOG_LINE << "Skipping frames from " << next_frame_idx_ << " to " << frame_id << std::endl;
  for (int i = next_frame_idx_; i < frame_id; i++) {
    ReadNextFrame(nullptr, false);
  }
  assert(next_frame_idx_ == frame_id);
}

bool FramesDecoderBase::ReadFlushFrame(uint8_t *data, bool copy_to_output) {
  if (avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_) < 0) {
    flush_state_ = false;
    return false;
  }

  if (copy_to_output) {
    CopyToOutput(data);
  }

  LOG_LINE << "Read frame (ReadFlushFrame), index " << next_frame_idx_ << " timestamp "
           << std::setw(5) << av_state_->frame_->pts << ", current copy " << copy_to_output
           << std::endl;
  ++next_frame_idx_;

  // TODO(awolant): Figure out how to handle this during index building
  // Or when NumFrames in unavailible
  if (next_frame_idx_ >= NumFrames()) {
    next_frame_idx_ = -1;
    LOG_LINE << "Next frame index out of bounds, setting to -1" << std::endl;
  }

  return true;
}

bool FramesDecoderBase::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  LOG_LINE << "ReadNextFrame: frame_idx=" << next_frame_idx_
            << " copy=" << copy_to_output
            << " flush=" << flush_state_ << std::endl;
  if (!flush_state_) {
    if (ReadRegularFrame(data, copy_to_output)) {
      return true;
    }
  }
  return ReadFlushFrame(data, copy_to_output);
}

}  // namespace dali
