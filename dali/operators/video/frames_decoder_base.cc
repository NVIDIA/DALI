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

#include "dali/operators/video/frames_decoder_base.h"
#include <cstring>
#include <iomanip>
#include <memory>
#include "dali/core/error_handling.h"
#include "dali/core/small_vector.h"
#include "dali/operators/video/video_utils.h"

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

int read_memory_video_file(void *data_ptr, uint8_t *av_io_buffer, int av_io_buffer_size) {
  MemoryVideoFile *memory_video_file = static_cast<MemoryVideoFile *>(data_ptr);
  return memory_video_file->Read(av_io_buffer, av_io_buffer_size);
}

int64_t seek_memory_video_file(void *data_ptr, int64_t new_position, int origin) {
  MemoryVideoFile *memory_video_file = static_cast<MemoryVideoFile *>(data_ptr);
  return memory_video_file->Seek(new_position, origin);
}

}  // namespace detail

int FramesDecoderBase::OpenFile(const std::string& filename) {
  LOG_LINE << "Opening file " << filename << std::endl;
  ctx_.reset(avformat_alloc_context());
  DALI_ENFORCE(ctx_, "Could not alloc avformat context");

  // avformat_open_input parses the filename through avio's URL layer, which treats a colon as a
  // protocol separator (e.g. a relative path like "clip:01.mp4" would be parsed as protocol
  // "clip", which doesn't exist, instead of a plain filename). Prefixing with the "file:"
  // protocol forces it to always be treated as a plain filesystem path, regardless of what
  // characters it contains, for both relative and absolute paths.
  std::string url = "file:" + filename;
  int ret = avformat_open_input(&ctx_, url.c_str(), nullptr, nullptr);
  if (ret < 0) {
    ctx_.reset();
  }
  return ret;
}

int FramesDecoderBase::OpenMemoryFile(MemoryVideoFile &memory_video_file) {
  LOG_LINE << "Opening memory file" << std::endl;
  ctx_.reset(avformat_alloc_context());
  DALI_ENFORCE(ctx_, "Could not alloc avformat context");

  static constexpr int DEFAULT_AV_BUFFER_SIZE = (1 << 15);
  uint8_t* buffer = static_cast<uint8_t*>(av_malloc(DEFAULT_AV_BUFFER_SIZE));
  DALI_ENFORCE(buffer, "Could not alloc avio context buffer");

  auto avio_ctx = avio_alloc_context(
    buffer,
    DEFAULT_AV_BUFFER_SIZE,
    0,
    &memory_video_file,
    detail::read_memory_video_file,
    nullptr,
    detail::seek_memory_video_file);

  if (!avio_ctx) {
    av_freep(&buffer);
    DALI_FAIL("Could not alloc avio context");
  }

  ctx_->pb = avio_ctx;

  int ret = avformat_open_input(&ctx_, "", nullptr, nullptr);
  if (ret < 0) {
    DestroyAvObject(&avio_ctx);
    ctx_.reset();
  }
  return ret;
}

int64_t FramesDecoderBase::NumFrames() {
  if (num_frames_ >= 0) {
    return num_frames_;
  }

  if (ctx_->streams[stream_id_]->nb_frames > 0) {
    num_frames_ = ctx_->streams[stream_id_]->nb_frames;
    return num_frames_;
  }

  ParseNumFrames();
  return num_frames_;
}

std::string FramesDecoderBase::GetAllStreamInfo() const {
  std::stringstream ss;
  ss << "Number of streams: " << ctx_->nb_streams << std::endl;
  for (size_t i = 0; i < ctx_->nb_streams; ++i) {
    ss << "Stream " << i << ": " << ctx_->streams[i]->codecpar->codec_type << std::endl;
    ss << "  Codec ID: " << ctx_->streams[i]->codecpar->codec_id << " ("
       << avcodec_get_name(ctx_->streams[i]->codecpar->codec_id) << ")" << std::endl;
    ss << "  Codec Type: " << ctx_->streams[i]->codecpar->codec_type << std::endl;
    ss << "  Format: " << ctx_->streams[i]->codecpar->format << std::endl;
    ss << "  Width: " << ctx_->streams[i]->codecpar->width << std::endl;
    ss << "  Height: " << ctx_->streams[i]->codecpar->height << std::endl;
    ss << "  Sample Rate: " << ctx_->streams[i]->codecpar->sample_rate << std::endl;
    ss << "  Bit Rate: " << ctx_->streams[i]->codecpar->bit_rate << std::endl;
  }
  return ss.str();
}

bool FramesDecoderBase::SelectVideoStream(int stream_id) {
  if (stream_id < 0) {
    LOG_LINE << "Finding video stream" << std::endl;
    stream_id = av_find_best_stream(ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_id == AVERROR_STREAM_NOT_FOUND) {
      // Some containers (e.g. MPEG-PS/TS) don't declare stream codec types in a header that's
      // available right after avformat_open_input -- they only become known once packets are
      // probed. Fall back to a full probe and retry before giving up.
      if (avformat_find_stream_info(ctx_, nullptr) >= 0) {
        stream_id = av_find_best_stream(ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
      }
      if (stream_id == AVERROR_STREAM_NOT_FOUND) {
        DALI_WARN(make_string("Could not find a valid video stream in a file in ", Filename()));
        return false;
      }
    }
  }
  if (stream_id < 0 || stream_id >= static_cast<int>(ctx_->nb_streams)) {
    LOG_LINE << "Invalid stream id: " << stream_id << std::endl;
    return false;
  }
  stream_id_ = stream_id;
  codec_params_ = ctx_->streams[stream_id_]->codecpar;
  LOG_LINE << "Selecting stream " << stream_id_
           << " (codec_id=" << codec_params_->codec_id
           << ", codec_type=" << codec_params_->codec_type
           << ", format=" << codec_params_->format
           << ", width=" << codec_params_->width
           << ", height=" << codec_params_->height
           << ", sample_rate=" << codec_params_->sample_rate
           << ", bit_rate=" << codec_params_->bit_rate << ")" << std::endl;

  assert(codec_params_->codec_type != AVMEDIA_TYPE_NB);
  switch (codec_params_->codec_type) {
    case AVMEDIA_TYPE_UNKNOWN:  // if unknown, we can't determine if it's a video stream
    case AVMEDIA_TYPE_VIDEO:
      break;
    case AVMEDIA_TYPE_AUDIO:  // fall through
    case AVMEDIA_TYPE_DATA:   // fall through
    case AVMEDIA_TYPE_SUBTITLE:  // fall through
    case AVMEDIA_TYPE_ATTACHMENT:  // fall through
    default:
      LOG_LINE << "Stream " << stream_id << " is not a video stream" << std::endl;
      codec_params_ = nullptr;
      stream_id_ = -1;
      return false;
  }
  LOG_LINE << "Selected stream " << stream_id << " with codec "
           << avcodec_get_name(codec_params_->codec_id) << " ("
           << codec_params_->codec_id << ")" << std::endl;
  if (!CheckDimensions())
    return false;

  next_frame_idx_ = 0;
  can_seek_ = true;
  return true;
}

bool FramesDecoderBase::CheckDimensions() {
  if (Height() == 0 || Width() == 0) {
    if (avformat_find_stream_info(ctx_, nullptr) < 0) {
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

FramesDecoderBase::FramesDecoderBase(const std::string &filename, DALIImageType image_type) {
  av_log_set_level(AV_LOG_ERROR);
  filename_ = filename;
  DALI_ENFORCE(image_type == DALI_YCbCr || image_type == DALI_RGB,
               make_string("Invalid image type: ", image_type));
  image_type_ = image_type;
  int ret = OpenFile(filename);
  if (ret < 0) {
    DALI_WARN(make_string("Failed to open video file \"", Filename(), "\", due to ",
                          av_error_string(ret)));
    return;
  }

  packet_.reset(av_packet_alloc());
  DALI_ENFORCE(packet_, "Could not allocate av packet");

  is_valid_ = true;
  can_seek_ = true;
  next_frame_idx_ = 0;
}

FramesDecoderBase::FramesDecoderBase(const char *memory_file, size_t memory_file_size, std::string_view source_info, DALIImageType image_type) {
  av_log_set_level(AV_LOG_ERROR);
  filename_ = source_info;
  DALI_ENFORCE(image_type == DALI_YCbCr || image_type == DALI_RGB,
               make_string("Invalid image type: ", image_type));
  image_type_ = image_type;
  memory_video_file_ = std::make_unique<MemoryVideoFile>(memory_file, memory_file_size);
  int ret = OpenMemoryFile(*memory_video_file_);
  if (ret < 0) {
    DALI_WARN(make_string("Failed to open video file from memory buffer due to: ",
                          av_error_string(ret)));
    return;
  }

  packet_.reset(av_packet_alloc());
  DALI_ENFORCE(packet_, "Could not allocate av packet");

  is_valid_ = true;
  can_seek_ = true;
  next_frame_idx_ = 0;
}

void FramesDecoderBase::ParseNumFrames() {
  num_frames_ = 0;
  while (true) {
    int ret = av_read_frame(ctx_, packet_);
    auto packet = AVPacketScope(packet_, av_packet_unref);
    if (ret != 0) {
      break;  // End of file
    }

    if (packet->stream_index != stream_id_) {
      continue;
    }
    ++num_frames_;
  }
  Reset();
}

namespace {

/**
 * @brief Checks for an Annex-B NAL unit start code (00 00 01 or 00 00 00 01) at `buf`.
 *
 * @param buf Pointer to the position to check.
 * @param end End of the buffer (exclusive), to avoid reading out of bounds.
 * @return The length of the start code (3 or 4), or 0 if no start code is present at `buf`.
 */
inline int nal_start_code_length(const uint8_t *buf, const uint8_t *end) {
  if (buf + 4 <= end && buf[0] == 0 && buf[1] == 0 && buf[2] == 0 && buf[3] == 1) {
    return 4;
  }
  if (buf + 3 <= end && buf[0] == 0 && buf[1] == 0 && buf[2] == 1) {
    return 3;
  }
  return 0;
}

/**
 * @brief Locates the next NAL unit starting at `pos`, using the stream's NAL unit framing.
 *
 * H.264/HEVC packets use one of two framings, fixed for the whole stream:
 * - AVCC/ISO (MP4/MOV/Matroska): each NAL unit is prefixed with its big-endian length, stored on
 *   `nal_length_size` bytes (1, 2 or 4, as recorded in the AVCDecoderConfigurationRecord /
 *   HEVCDecoderConfigurationRecord extradata -- see GetNalLengthSize).
 * - Annex-B (raw elementary streams, AVI, MPEG-PS/TS, ...): NAL units are delimited by start
 *   codes (00 00 01 or 00 00 00 01); signaled here by `nal_length_size == 0`.
 *
 * The framing must be known up front, it can't be guessed per NAL unit: an AVCC length prefix
 * can be byte-for-byte identical to a start code (e.g. any 4-byte length in [256, 511] is
 * "00 00 01 XX", and a length of 1 is "00 00 00 01").
 *
 * @param pos [in/out] Position to start searching from; advanced past the located NAL unit.
 * @param end End of the packet buffer (exclusive).
 * @param nal_length_size Size of the AVCC/ISO length prefix, or 0 for Annex-B.
 * @param nal_data [out] Set to the first byte of the NAL unit (after any start code or
 * length prefix).
 * @param nal_size [out] Set to the size, in bytes, of the NAL unit.
 * @return true if a NAL unit was located, false if there isn't enough data left in the packet.
 */
inline bool find_next_nal_unit(const uint8_t *&pos, const uint8_t *end, int nal_length_size,
                               const uint8_t *&nal_data, uint32_t &nal_size) {
  if (nal_length_size > 0) {
    if (end - pos < nal_length_size) {
      return false;
    }
    uint32_t length = 0;
    for (int i = 0; i < nal_length_size; i++) {
      length = (length << 8) | pos[i];
    }
    nal_data = pos + nal_length_size;
    if (length > static_cast<size_t>(end - nal_data)) {
      return false;
    }
    nal_size = length;
    pos = nal_data + nal_size;
    return true;
  }

  // Annex-B: skip to the next start code; the NAL unit runs from there up to the following
  // start code (or the end of the packet).
  int sc_len = 0;
  while (pos < end && (sc_len = nal_start_code_length(pos, end)) == 0) {
    ++pos;
  }
  if (sc_len == 0) {
    return false;
  }
  nal_data = pos + sc_len;
  const uint8_t *next = nal_data;
  while (next < end && nal_start_code_length(next, end) == 0) {
    ++next;
  }
  nal_size = static_cast<uint32_t>(next - nal_data);
  pos = next;
  return true;
}

}  // namespace

namespace detail {

int GetNalLengthSize(AVCodecID codec_id, const uint8_t *extradata, int extradata_size) {
  // Both AVCDecoderConfigurationRecord (ISO/IEC 14496-15, 5.3.3.1) and
  // HEVCDecoderConfigurationRecord (8.3.3.1) start with configurationVersion = 1, while Annex-B
  // extradata starts with a start code (00 00 01 / 00 00 00 01). lengthSizeMinusOne is held in the
  // two low bits of byte 4 (avcC) or byte 21 (hvcC).
  if (extradata == nullptr) {
    return 0;
  }
  if (codec_id == AV_CODEC_ID_H264 && extradata_size > 4 && extradata[0] == 1) {
    return (extradata[4] & 0x03) + 1;
  }
  // Some early HEVC muxers wrote hvcC with configurationVersion = 0, so -- like FFmpeg's HEVC
  // decoder -- treat anything that doesn't start like an Annex-B start code as hvcC.
  if (codec_id == AV_CODEC_ID_HEVC && extradata_size > 21 &&
      (extradata[0] != 0 || extradata[1] != 0 || extradata[2] > 1)) {
    return (extradata[21] & 0x03) + 1;
  }
  return 0;
}

bool HasKeyframeNalUnit(AVCodecID codec_id, const uint8_t *data, int size,
                        int nal_length_size) {
  if (data == nullptr || size <= 0) {
    return false;
  }
  const uint8_t *pos = data;
  const uint8_t *end = data + size;
  const uint8_t *nal_data;
  uint32_t nal_size;
  while (find_next_nal_unit(pos, end, nal_length_size, nal_data, nal_size)) {
    if (nal_size == 0) {
      continue;
    }
    if (codec_id == AV_CODEC_ID_H264) {
      // In H.264, the NAL unit type is in the lower 5 bits. Type 5 is an IDR (Instantaneous
      // Decoding Refresh) slice, which clears all reference buffers.
      if ((nal_data[0] & 0x1F) == 5) {
        return true;
      }
    } else if (codec_id == AV_CODEC_ID_HEVC) {
      // In HEVC, the NAL unit type is in bits 1-6 of the first byte. Types 16-21 are IRAP
      // (Intra Random Access Point) pictures.
      uint8_t nal_unit_type = (nal_data[0] >> 1) & 0x3F;
      if (nal_unit_type >= 16 && nal_unit_type <= 21) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace detail

void FramesDecoderBase::BuildIndex() {
  if (HasIndex()) {
    return;
  }

  // MPEG-PS only stamps a timestamp on some packets (typically the first packet of each PES
  // unit), which the frame index built below can't support reliably (it needs an exact,
  // collision-free timestamp for every frame). Reject it here, rather than in
  // SelectVideoStream(), so the rejection only applies to callers that actually need a seek
  // index (e.g. experimental.readers.video): sequential-decode-only operators such as
  // experimental.decoders.video and experimental.inputs.video never call BuildIndex() and can
  // still open and decode MPEG-PS content. MPEG-TS is unaffected (its packets carry
  // PCR-derived timestamps far more consistently) and is not rejected by this check.
  if (!strcmp(ctx_->iformat->name, "mpeg")) {
    DALI_FAIL(make_string(
        "Video file \"", Filename(), "\" is MPEG-PS (MPEG-2 Program Stream), which does not "
        "support building a frame-accurate seek index: this container format only stamps a "
        "timestamp on some packets. Remux the file to MP4 or MKV (e.g. `ffmpeg -i in.mpeg -c "
        "copy out.mp4`) to use it where a seek index is required."));
  }

  // Initialize frame index
  index_.index.clear();
  index_.filename = Filename();
  index_.timebase = ctx_->streams[stream_id_]->time_base;

  // The NAL unit framing of H.264/HEVC packets (Annex-B start codes vs. AVCC/ISO length
  // prefixes) is a property of the whole stream, signaled by its extradata. It has to be
  // determined here, because av_read_frame returns packets exactly as stored in the container:
  // the *_mp4toannexb bitstream filter is only applied later, on the GPU decode path.
  const AVCodecParameters *codecpar = ctx_->streams[stream_id_]->codecpar;
  const AVCodecID codec_id = codecpar->codec_id;
  const int nal_length_size =
      detail::GetNalLengthSize(codec_id, codecpar->extradata, codecpar->extradata_size);

  // Track the position of the last keyframe seen
  int last_keyframe = -1;
  int frame_count = 0;
  num_frames_ = 0;

  while (true) {
    // Read the next frame from the video
    int ret = av_read_frame(ctx_, packet_);
    auto packet = AVPacketScope(packet_, av_packet_unref);
    if (ret != 0) {
      LOG_LINE << "End of file reached after " << frame_count << " frames" << std::endl;
      break;  // Just break when we hit EOF instead of trying to seek back
    }

    frame_count++;

    // Skip packets from other streams (e.g. audio)
    if (packet->stream_index != stream_id_) {
      continue;
    }

    IndexEntry entry;
    entry.is_keyframe = false;  // Default to false, set true only if confirmed

    // Check if this packet contains a keyframe
    if (packet->flags & AV_PKT_FLAG_KEY) {
      LOG_LINE << "Found potential keyframe at frame " << index_.size() << std::endl;

      // Special handling for H.264 and HEVC formats: parse the packet's NAL units (Network
      // Abstraction Layer, the basic unit of encoded video) to verify that this is actually a
      // keyframe.
      if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC) {
        entry.is_keyframe =
            detail::HasKeyframeNalUnit(codec_id, packet->data, packet->size, nal_length_size);
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

    if (entry.pts < 0) {
      LOG_LINE << "Negative timestamp: " << entry.pts << ", skipping" << std::endl;
      continue;
    }

    // Update last keyframe position if this is a keyframe
    if (entry.is_keyframe) {
      last_keyframe = index_.size();
    }
    entry.last_keyframe_id = last_keyframe;

    // Regular frame, not a flush frame
    entry.is_flush_frame = false;
    index_.index.push_back(entry);
    ++num_frames_;
  }

  LOG_LINE << "Index building complete. Total frames: " << index_.size() << std::endl;

  DALI_ENFORCE(index_.index.size() > 0,
               make_string("No valid frames found in video file \"", Filename(), "\""));

  // Mark last frame as flush frame
  index_.index.back().is_flush_frame = true;

  // Sort frames by presentation timestamp
  // This is needed because frames may be stored out of order in the container
  std::sort(index_.index.begin(), index_.index.end(),
            [](const IndexEntry &a, const IndexEntry &b) { return a.pts < b.pts; });

  // After sorting, we need to update last_keyframe_id references
  std::vector<int> keyframe_positions;
  for (size_t i = 0; i < index_.size(); i++) {
    if (index_[i].is_keyframe) {
      keyframe_positions.push_back(i);
    }
  }

  if (keyframe_positions.empty()) {
    keyframe_positions.push_back(0);
  }

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

bool FramesDecoderBase::AvSeekFrame(int64_t timestamp, int frame_id) {
  if (!can_seek_) {
    LOG_LINE << "Not seekable, returning directly" << std::endl;
    return false;
  }

  can_seek_ =
      av_seek_frame(ctx_, stream_id_, timestamp, AVSEEK_FLAG_BACKWARD) >= 0;
  if (!can_seek_)
    return false;

  LOG_LINE << "Seeked to frame " << frame_id << std::endl;
  Flush();

  next_frame_idx_ = frame_id;
  return true;
}

void FramesDecoderBase::Reset() {
  LOG_LINE << "Reset: Reopening stream." << std::endl;
  int stream_id = stream_id_;

  int ret = -1;
  if (memory_video_file_) {
    memory_video_file_->Seek(0, SEEK_SET);
    ret = OpenMemoryFile(*memory_video_file_);
    DALI_ENFORCE(ret >= 0,
                 make_string("Could not open video file from memory buffer due to: ",
                             av_error_string(ret)));
  } else {
    ret = OpenFile(Filename());
    DALI_ENFORCE(ret >= 0,
                 make_string("Could not open video file \"", Filename(),
                    "\" due to: ", av_error_string(ret)));
  }

  is_valid_ = true;
  can_seek_ = true;
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
      const auto &current_frame = index_[next_frame_idx_];
      const auto &requested_frame = index_[frame_id];
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
        auto &keyframe_entry = index_[keyframe_id];
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
    uint8_t* out_frame_start = data + ptrdiff_t(i) * FrameSizeBytes();
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
    int64_t pts_0 = index_[0].pts;
    for (auto& [frame_idx, i] : frame_ids) {
      if (frame_idx >= 0) {
        out_timestamps[i] = TimestampToSeconds(GetTimebase(), index_[frame_idx].pts - pts_0);
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
