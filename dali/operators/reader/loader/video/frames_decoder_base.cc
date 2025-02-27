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

int AvState::OpenFile(const std::string& filename) {
  LOG_LINE << "Opening file " << filename << std::endl;
  CloseInput();
  ctx_ = avformat_alloc_context();
  DALI_ENFORCE(ctx_, "Could not alloc avformat context");
  int ret = avformat_open_input(&ctx_, filename.c_str(), nullptr, nullptr);
  if (ret < 0) {
    CloseInput();
  }
  return ret;
}

int AvState::OpenMemoryFile(MemoryVideoFile &memory_video_file) {
  LOG_LINE << "Opening memory file" << std::endl;
  CloseInput();
  ctx_ = avformat_alloc_context();
  DALI_ENFORCE(ctx_, "Could not alloc avformat context");

  static constexpr int DEFAULT_AV_BUFFER_SIZE = (1 << 15);
  uint8_t *avio_buffer = static_cast<uint8_t *>(av_malloc(DEFAULT_AV_BUFFER_SIZE));
  DALI_ENFORCE(avio_buffer, "Could not allocate avio buffer");

  auto avio_ctx = avio_alloc_context(
    avio_buffer,
    DEFAULT_AV_BUFFER_SIZE,
    0,
    &memory_video_file,
    detail::read_memory_video_file,
    nullptr,
    detail::seek_memory_video_file);
  if (avio_ctx == nullptr) {
    av_free(avio_buffer);
    DALI_FAIL("Could not allocate avio context");
  }

  ctx_->pb = avio_ctx;
  int ret = avformat_open_input(&ctx_, "", nullptr, nullptr);
  if (ret < 0) {
    CloseInput();
  }
  return ret;
}

using AVPacketScope = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;

int64_t FramesDecoderBase::NumFrames() const {
  if (num_frames_ >= 0) {
    return num_frames_;
  }

  if (!index_.empty()) {
    return index_.size();
  }

  return av_state_->ctx_->streams[av_state_->stream_id_]->nb_frames;
}

void FramesDecoderBase::InitAvCodecContext() {
  av_state_->codec_ctx_ = avcodec_alloc_context3(av_state_->codec_);
  DALI_ENFORCE(av_state_->codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(av_state_->codec_ctx_, av_state_->codec_params_);
  DALI_ENFORCE(ret >= 0, make_string("Could not fill the codec based on parameters: ",
                                     detail::av_error_string(ret)));

  av_state_->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state_->packet_, "Could not allocate av packet");
}

bool FramesDecoderBase::OpenAvCodec() {
  int ret = avcodec_open2(av_state_->codec_ctx_, av_state_->codec_, nullptr);
  if (ret != 0) {
    DALI_WARN(make_string("Could not initialize codec context: ", detail::av_error_string(ret)));
    return false;
  }
  av_state_->frame_ = av_frame_alloc();
  if (av_state_->frame_ == nullptr) {
    DALI_WARN("Could not allocate the av frame");
    return false;
  }
  return true;
}

std::string FramesDecoderBase::GetAllStreamInfo() const {
  std::stringstream ss;
  ss << "Number of streams: " << av_state_->ctx_->nb_streams << std::endl;
  for (size_t i = 0; i < av_state_->ctx_->nb_streams; ++i) {
    ss << "Stream " << i << ": " << av_state_->ctx_->streams[i]->codecpar->codec_type << std::endl;
    ss << "  Codec ID: " << av_state_->ctx_->streams[i]->codecpar->codec_id << " ("
       << avcodec_get_name(av_state_->ctx_->streams[i]->codecpar->codec_id) << ")" << std::endl;
    ss << "  Codec Type: " << av_state_->ctx_->streams[i]->codecpar->codec_type << std::endl;
    ss << "  Format: " << av_state_->ctx_->streams[i]->codecpar->format << std::endl;
    ss << "  Width: " << av_state_->ctx_->streams[i]->codecpar->width << std::endl;
    ss << "  Height: " << av_state_->ctx_->streams[i]->codecpar->height << std::endl;
    ss << "  Sample Rate: " << av_state_->ctx_->streams[i]->codecpar->sample_rate << std::endl;
    ss << "  Bit Rate: " << av_state_->ctx_->streams[i]->codecpar->bit_rate << std::endl;
  }
  return ss.str();
}

bool FramesDecoderBase::SelectVideoStream(int stream_id, bool require_available_avcodec) {
  if (stream_id < 0 || stream_id >= static_cast<int>(av_state_->ctx_->nb_streams)) {
    return false;
  }
  av_state_->stream_id_ = stream_id;
  av_state_->codec_params_ = av_state_->ctx_->streams[stream_id]->codecpar;

  LOG_LINE << "Selecting stream " << stream_id
           << " (codec_id=" << av_state_->codec_params_->codec_id
           << ", codec_type=" << av_state_->codec_params_->codec_type
           << ", format=" << av_state_->codec_params_->format
           << ", width=" << av_state_->codec_params_->width
           << ", height=" << av_state_->codec_params_->height
           << ", sample_rate=" << av_state_->codec_params_->sample_rate
           << ", bit_rate=" << av_state_->codec_params_->bit_rate << ")" << std::endl;

  if (require_available_avcodec) {
    if ((av_state_->codec_ = avcodec_find_decoder(av_state_->codec_params_->codec_id)) == nullptr) {
      LOG_LINE << "No decoder found for codec "
               << avcodec_get_name(av_state_->codec_params_->codec_id)
               << " (codec_id=" << av_state_->codec_params_->codec_id << ")" << std::endl;
      return false;
    }
    if (av_state_->codec_->type != AVMEDIA_TYPE_VIDEO) {
      LOG_LINE << "Stream " << stream_id << " is not a video stream" << std::endl;
      av_state_->codec_ = nullptr;
      av_state_->codec_params_ = nullptr;
      av_state_->stream_id_ = -1;
      return false;
    }
  }

  assert(av_state_->codec_params_->codec_type != AVMEDIA_TYPE_NB);
  switch (av_state_->codec_params_->codec_type) {
    case AVMEDIA_TYPE_UNKNOWN:  // if unknown, we can't determine if it's a video stream
    case AVMEDIA_TYPE_VIDEO:
      break;
    case AVMEDIA_TYPE_AUDIO:  // fall through
    case AVMEDIA_TYPE_DATA:   // fall through
    case AVMEDIA_TYPE_SUBTITLE:  // fall through
    case AVMEDIA_TYPE_ATTACHMENT:  // fall through
    default:
      LOG_LINE << "Stream " << stream_id << " is not a video stream" << std::endl;
      av_state_->codec_ = nullptr;
      av_state_->codec_params_ = nullptr;
      av_state_->stream_id_ = -1;
      return false;
  }
  LOG_LINE << "Selected stream " << stream_id << " with codec "
           << avcodec_get_name(av_state_->codec_params_->codec_id) << " ("
           << av_state_->codec_params_->codec_id << ")" << std::endl;
  return true;
}

bool FramesDecoderBase::FindVideoStream(bool require_available_avcodec) {
  LOG_LINE << "Finding video stream (require_available_avcodec=" << require_available_avcodec << ")"
           << std::endl;
  if (require_available_avcodec) {
    size_t i = 0;
    LOG_LINE << "Checking " << av_state_->ctx_->nb_streams << " streams" << std::endl;
    for (i = 0; i < av_state_->ctx_->nb_streams; ++i) {
      LOG_LINE << "Checking stream " << i << std::endl;
      if (SelectVideoStream(i, true)) {
        LOG_LINE << "Found video stream " << i << std::endl;
        break;
      }
    }
    if (av_state_->stream_id_ == -1) {
      LOG_LINE << "Could not find a valid video stream in a file " << Filename() << std::endl;
      return false;
    }
  } else {
    int stream_id = av_find_best_stream(av_state_->ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_id == AVERROR_STREAM_NOT_FOUND) {
      LOG_LINE << "No valid video stream found" << std::endl;
      return false;
    }
    if (!SelectVideoStream(stream_id, false)) {
      LOG_LINE << "Could not select video stream " << stream_id << std::endl;
      return false;
    }
  }
  return CheckDimensions();
}

bool FramesDecoderBase::CheckDimensions() {
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

FramesDecoderBase::FramesDecoderBase(const std::string &filename, bool build_index,
                                     bool init_codecs)
    : av_state_(std::make_unique<AvState>()) {
  av_log_set_level(AV_LOG_ERROR);
  filename_ = filename;

  int ret = av_state_->OpenFile(filename);
  if (ret < 0) {
    DALI_WARN(make_string("Failed to open video file \"", Filename(), "\", due to ",
                          detail::av_error_string(ret)));
    return;
  }

  if (!FindVideoStream(init_codecs)) {
    DALI_WARN(make_string("Could not find a valid video stream in a file ", Filename(),
                          ". Streams available: ", GetAllStreamInfo()));
    return;
  }

  InitAvCodecContext();
  if (init_codecs) {
    if (!OpenAvCodec()) {
      is_valid_ = false;
      return;
    }
  }

  if (build_index) {
    LOG_LINE << "Building index" << std::endl;
    BuildIndex();
  } else if (NumFrames() == 0) {
    ParseNumFrames();
    LOG_LINE << "Parsed number of frames: " << NumFrames() << std::endl;
  }

  is_valid_ = true;
  can_seek_ = true;
  next_frame_idx_ = 0;
}

FramesDecoderBase::FramesDecoderBase(const char *memory_file, int memory_file_size,
                                     bool build_index, bool init_codecs, int num_frames,
                                     std::string_view source_info)
    : av_state_(std::make_unique<AvState>()) {
  av_log_set_level(AV_LOG_ERROR);

  filename_ = source_info;
  num_frames_ = num_frames;

  memory_video_file_ = std::make_unique<MemoryVideoFile>(memory_file, memory_file_size);
  int ret = av_state_->OpenMemoryFile(*memory_video_file_);
  if (ret < 0) {
    DALI_WARN(make_string("Failed to open video file from memory buffer due to: ",
                          detail::av_error_string(ret)));
    return;
  }

  if (!FindVideoStream(init_codecs)) {
    DALI_WARN(
        make_string("Could not find a valid video stream in the memory buffer. Streams available: ",
                    GetAllStreamInfo()));
    return;
  }

  InitAvCodecContext();
  if (init_codecs) {
    if (!OpenAvCodec()) {
      is_valid_ = false;
      return;
    }
  }

  if (build_index) {
    LOG_LINE << "Building index" << std::endl;
    BuildIndex();
  } else if (NumFrames() == 0) {
    ParseNumFrames();
    LOG_LINE << "Parsed number of frames: " << NumFrames() << std::endl;
  }

  is_valid_ = true;
  can_seek_ = true;
  next_frame_idx_ = 0;
}

void FramesDecoderBase::ParseNumFrames() {
  CountFrames();
  Reset();
}

void FramesDecoderBase::CountFrames() {
  num_frames_ = 0;
  while (true) {
    int ret = av_read_frame(av_state_->ctx_, av_state_->packet_);
    auto packet = AVPacketScope(av_state_->packet_, av_packet_unref);
    if (ret != 0) {
      break;  // End of file
    }

    if (packet->stream_index != av_state_->stream_id_) {
      continue;
    }
    ++num_frames_;
  }
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
      last_keyframe = index_.size();
    }
    entry.last_keyframe_id = last_keyframe;

    // Regular frame, not a flush frame
    entry.is_flush_frame = false;
    index_.push_back(entry);
  }

  LOG_LINE << "Index building complete. Total frames: " << index_.size() << std::endl;

  DALI_ENFORCE(!index_.empty(),
               make_string("No valid frames found in video file \"", Filename(), "\""));

  // Mark last frame as flush frame
  index_.back().is_flush_frame = true;

  // Sort frames by presentation timestamp
  // This is needed because frames may be stored out of order in the container
  std::sort(index_.begin(), index_.end(),
            [](const IndexEntry &a, const IndexEntry &b) { return a.pts < b.pts; });

  // After sorting, we need to update last_keyframe_id references
  std::vector<int> keyframe_positions;
  for (size_t i = 0; i < index_.size(); i++) {
    if (index_[i].is_keyframe) {
      keyframe_positions.push_back(i);
    }
  }

  DALI_ENFORCE(!keyframe_positions.empty(),
               make_string("No keyframes found in video file \"", Filename(), "\""));

  // Update last_keyframe_id for each frame after sorting
  for (size_t i = 0; i < index_.size(); i++) {
    // Find the last keyframe that comes before or at this frame
    auto it = std::upper_bound(keyframe_positions.begin(), keyframe_positions.end(), i);
    if (it == keyframe_positions.begin()) {
      index_[i].last_keyframe_id = 0;  // First keyframe
    } else {
      index_[i].last_keyframe_id = *(--it);
    }
  }

  // Detect if video has variable frame rate (VFR)
  DetectVariableFrameRate();
  Reset();
}

void FramesDecoderBase::DetectVariableFrameRate() {
  is_vfr_ = false;
  if (index_.size() > 3) {
    int64_t pts_step = index_[1].pts - index_[0].pts;
    for (size_t i = 2; i < index_.size(); i++) {
      if (index_[i].pts - index_[i-1].pts != pts_step) {
        is_vfr_ = true;
        break;
      }
    }
  }
}

void FramesDecoderBase::CopyToOutput(uint8_t *data) {
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

bool FramesDecoderBase::ReadRegularFrame(uint8_t *data) {
  int ret = -1;
  bool copy_to_output = data != nullptr;
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

    LOG_LINE << (copy_to_output ? "Read" : "Skip") << " frame (ReadRegularFrame), index "
             << next_frame_idx_ << ", timestamp " << std::setw(5) << av_state_->frame_->pts
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

bool FramesDecoderBase::AvSeekFrame(int64_t timestamp, int frame_id) {
  if (!can_seek_) {
    LOG_LINE << "Not seekable, returning directly" << std::endl;
    return false;
  }

  if (av_state_->codec_) {
    LOG_LINE << "Flushing codec" << std::endl;
    avcodec_flush_buffers(av_state_->codec_ctx_);
  }

  can_seek_ =
      av_seek_frame(av_state_->ctx_, av_state_->stream_id_, timestamp, AVSEEK_FLAG_FRAME) >= 0;
  if (!can_seek_)
    return false;

  LOG_LINE << "Seeked to frame " << frame_id << " flush_state_=" << flush_state_ << std::endl;
  // Seeking clears av buffers, so reset flush state info
  if (flush_state_) {
    LOG_LINE << "Flushing frames" << std::endl;
    while (ReadFlushFrame(nullptr)) {}
    flush_state_ = false;
  }

  next_frame_idx_ = frame_id;
  return true;
}

void FramesDecoderBase::Reset() {
  if (AvSeekFrame(0, 0)) {
    LOG_LINE << "Reset: Seeked to first frame." << std::endl;
    return;
  }

  LOG_LINE << "Reset: Failed to seek to first frame. Reopening stream." << std::endl;
  bool require_available_avcodec = av_state_->codec_ != nullptr;
  int stream_id = av_state_->stream_id_;

  int ret = -1;
  if (memory_video_file_) {
    memory_video_file_->Seek(0, SEEK_SET);
    ret = av_state_->OpenMemoryFile(*memory_video_file_);
    DALI_ENFORCE(ret >= 0,
                 make_string("Could not open video file from memory buffer due to: ",
                             detail::av_error_string(ret)));
  } else {
    ret = av_state_->OpenFile(Filename());
    DALI_ENFORCE(ret >= 0,
                 make_string("Could not open video file \"", Filename(),
                    "\" due to: ", detail::av_error_string(ret)));
  }

  SelectVideoStream(stream_id, require_available_avcodec);
  DALI_ENFORCE(CheckDimensions(), "Could not load video dimensions");
  InitAvCodecContext();
  if (require_available_avcodec) {
    OpenAvCodec();
  }
  flush_state_ = false;
  next_frame_idx_ = 0;
  can_seek_ = true;
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

  if (next_frame_idx_ < 0) {
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

bool FramesDecoderBase::ReadFlushFrame(uint8_t *data) {
  bool copy_to_output = data != nullptr;
  if (avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_) < 0) {
    flush_state_ = false;
    return false;
  }

  if (copy_to_output) {
    CopyToOutput(data);
  }

  LOG_LINE << (copy_to_output ? "Read" : "Skip") << "frame (ReadFlushFrame), index "
           << next_frame_idx_ << " timestamp " << std::setw(5) << av_state_->frame_->pts
           << ", copy_to_output=" << copy_to_output << std::endl;
  ++next_frame_idx_;

  // TODO(awolant): Figure out how to handle this during index building
  // Or when NumFrames in unavailible
  if (next_frame_idx_ >= NumFrames()) {
    next_frame_idx_ = -1;
    LOG_LINE << "Next frame index out of bounds, setting to -1" << std::endl;
  }

  return true;
}

bool FramesDecoderBase::ReadNextFrame(uint8_t *data) {
  LOG_LINE << (data != nullptr ? "Read" : "Skip") << " frame (ReadNextFrame), index "
           << next_frame_idx_ << " flush=" << flush_state_ << std::endl;
  if (!flush_state_) {
    if (ReadRegularFrame(data)) {
      return true;
    }
  }
  return ReadFlushFrame(data);
}

}  // namespace dali
