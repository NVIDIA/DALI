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

#include "dali/operators/reader/loader/video/frames_decoder.h"
#include <memory>
#include <iomanip>
#include "dali/core/error_handling.h"
#include "dali/core/util.h"


namespace dali {
int MemoryVideoFile::Read(unsigned char *buffer, int buffer_size) {
  int left_in_file = size_ - position_;
  if (left_in_file == 0) {
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
 * @param mode Chosen method of seeking. This argument changes how new_position is interpreted and how seeking is performed.
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
    DALI_FAIL(
      make_string(
        "Unsupported seeking method in FramesDecoder from memory file. Seeking method: ",
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

}   // namespace detail

using AVPacketScope = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;

const std::vector<AVCodecID> FramesDecoder::SupportedCodecs = {
  AVCodecID::AV_CODEC_ID_H264,
  AVCodecID::AV_CODEC_ID_HEVC,
  AVCodecID::AV_CODEC_ID_MPEG4
};

int64_t FramesDecoder::NumFrames() const {
    if (num_frames_.has_value()) {
      return num_frames_.value();
    }

    if (index_.has_value()) {
      return index_->size();
    }

    return av_state_->ctx_->streams[av_state_->stream_id_]->nb_frames;
}

void FramesDecoder::InitAvState(bool init_codecs) {
  av_state_->codec_ctx_ = avcodec_alloc_context3(av_state_->codec_);
  DALI_ENFORCE(av_state_->codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(av_state_->codec_ctx_, av_state_->codec_params_);
  DALI_ENFORCE(
    ret >= 0,
    make_string("Could not fill the codec based on parameters: ", detail::av_error_string(ret)));

  av_state_->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state_->packet_, "Could not allocate av packet");

  if (init_codecs) {
    ret = avcodec_open2(av_state_->codec_ctx_, av_state_->codec_, nullptr);
    DALI_ENFORCE(
      ret == 0,
      make_string("Could not initialize codec context: ", detail::av_error_string(ret)));

    av_state_->frame_ = av_frame_alloc();
    DALI_ENFORCE(av_state_->frame_, "Could not allocate the av frame");
  }
}

bool FramesDecoder::CheckCodecSupport() {
  return std::find(
    SupportedCodecs.begin(),
    SupportedCodecs.end(),
    av_state_->codec_params_->codec_id) != SupportedCodecs.end();
}

bool FramesDecoder::FindVideoStream(bool init_codecs) {
  if (init_codecs) {
    size_t i = 0;

    for (i = 0; i < av_state_->ctx_->nb_streams; ++i) {
      av_state_->codec_params_ = av_state_->ctx_->streams[i]->codecpar;
      av_state_->codec_ = avcodec_find_decoder(av_state_->codec_params_->codec_id);

      if (av_state_->codec_ == nullptr) {
        continue;
      }

      if (av_state_->codec_->type == AVMEDIA_TYPE_VIDEO) {
        av_state_->stream_id_ = i;
        break;
      }
    }

    if (i >= av_state_->ctx_->nb_streams) {
      DALI_WARN(make_string("Could not find a valid video stream in a file \"", Filename(), "\""));
      return false;
    }
  } else {
    av_state_->stream_id_ = av_find_best_stream(av_state_->ctx_, AVMEDIA_TYPE_VIDEO,
                                                -1, -1, nullptr, 0);

    LOG_LINE << "Best stream " << av_state_->stream_id_ << std::endl;
    if (av_state_->stream_id_ < 0) {
      DALI_WARN(make_string("Could not find a valid video stream in a file \"", Filename(), "\""));
      return false;
    }

    av_state_->codec_params_ = av_state_->ctx_->streams[av_state_->stream_id_]->codecpar;
  }
  if (Height() == 0 || Width() == 0) {
    if (avformat_find_stream_info(av_state_->ctx_, nullptr) < 0) {
      DALI_WARN(make_string("Could not find stream information in \"", Filename(), "\""));
      return false;
    }
    if (Height() == 0 || Width() == 0) {
      DALI_WARN("Couldn't load video size info.");
      return false;
    }
  }
  return true;
}

FramesDecoder::FramesDecoder(const std::string &filename)
    : av_state_(std::make_unique<AvState>()), filename_(filename) {

  av_log_set_level(AV_LOG_ERROR);

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
  if (!CheckCodecSupport()) {
    DALI_WARN(make_string("Unsupported video codec: ", CodecName(),
                          ". Supported codecs: h264, HEVC."));
    return;
  }
  InitAvState();
  BuildIndex();
  DetectVfr();
  is_valid_ = true;
}


FramesDecoder::FramesDecoder(const char *memory_file, int memory_file_size, bool build_index,
                             bool init_codecs, int num_frames, std::string_view source_info)
  : av_state_(std::make_unique<AvState>()),
    filename_(source_info),
    memory_video_file_(MemoryVideoFile(memory_file, memory_file_size)) {
  DALI_ENFORCE(init_codecs || !build_index,
               "FramesDecoder doesn't support index without CPU codecs");
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
    DALI_WARN(make_string("Failed to open video file \"", Filename(), "\", due to ",
                          detail::av_error_string(ret)));
    return;
  }

  if (!FindVideoStream(init_codecs || build_index)) {
    return;
  }
  if (!CheckCodecSupport()) {
    DALI_WARN(make_string("Unsupported video codec: \"", CodecName(),
                          "\". Supported codecs: h264, HEVC."));
    return;
  }
  InitAvState(init_codecs || build_index);

  // Number of frames is unknown and we do not plan to build the index
  if (NumFrames() == 0 && !build_index) {
    ParseNumFrames();
  }

  if (!build_index) {
    is_valid_ = true;
    return;
  }

  BuildIndex();
  DetectVfr();
  is_valid_ = true;
}

void FramesDecoder::CreateAvState(std::unique_ptr<AvState> &av_state, bool init_codecs) {
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
    DALI_ENFORCE(
      ret == 0,
      make_string(
        "Failed to open video file \"",
        Filename(),
        "\", due to ",
        detail::av_error_string(ret)));
    av_state->stream_id_ = av_find_best_stream(
      av_state->ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    av_state->codec_params_ = av_state->ctx_->streams[av_state->stream_id_]->codecpar;

    av_state->codec_ctx_ = avcodec_alloc_context3(av_state->codec_);
    DALI_ENFORCE(av_state->codec_ctx_, "Could not alloc av codec context");

    ret = avcodec_parameters_to_context(av_state->codec_ctx_, av_state->codec_params_);
    DALI_ENFORCE(
      ret >= 0,
      make_string("Could not fill the codec based on parameters: ", detail::av_error_string(ret)));

    av_state->packet_ = av_packet_alloc();
    DALI_ENFORCE(av_state->packet_, "Could not allocate av packet");
}

void FramesDecoder::ParseNumFrames() {
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

void FramesDecoder::CountFrames(AvState *av_state) {
  num_frames_ = 0;
  while (av_read_frame(av_state->ctx_, av_state->packet_) >= 0) {
    // We want to make sure that we call av_packet_unref in every iteration
    auto packet = AVPacketScope(av_state->packet_, av_packet_unref);

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
    if (iformat->read_seek == nullptr &&
        iformat->read_seek2 == nullptr)
      return false;
  } else {
    if (iformat->flags & (AVFMT_NOBINSEARCH | AVFMT_NOGENSEARCH))
      return false;
  }
  return true;
}

bool FramesDecoder::IsFormatSeekable() {
  if (!IsFormatSeekableHelper(av_state_->ctx_->iformat))
    return false;
  return av_state_->ctx_->pb->read_seek != nullptr;
}

void FramesDecoder::BuildIndex() {
  // TODO(awolant): Optimize this function for:
  //  - CFR
  //  - index present in the header

  index_ = vector<IndexEntry>();

  int last_keyframe = -1;
  while (ReadRegularFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = av_state_->frame_->flags & AV_FRAME_FLAG_KEY;
    entry.pts = av_state_->frame_->pts;
    entry.is_flush_frame = false;

    if (entry.is_keyframe) {
      last_keyframe = index_->size();
    }
    entry.last_keyframe_id = last_keyframe;
    index_->push_back(entry);
  }
  while (ReadFlushFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = false;
    entry.pts = av_state_->frame_->pts;
    entry.is_flush_frame = true;
    entry.last_keyframe_id = last_keyframe;
    index_->push_back(entry);
  }
  Reset();
}

void FramesDecoder::DetectVfr() {
  if (NumFrames() < 3) {
    is_vfr_ = false;
    return;
  }

  int pts_step = Index(1).pts - Index(0).pts;
  for (int frame_id = 2; frame_id < NumFrames(); ++frame_id) {
    if ((Index(frame_id).pts - Index(frame_id - 1).pts) != pts_step) {
      is_vfr_ = true;
      return;
    }
  }

  is_vfr_ = false;
}

void FramesDecoder::CopyToOutput(uint8_t *data) {
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

  DALI_ENFORCE(
    ret >= 0,
    make_string("Could not convert frame data to RGB: ", detail::av_error_string(ret)));
}

void FramesDecoder::LazyInitSwContext() {
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

bool FramesDecoder::ReadRegularFrame(uint8_t *data, bool copy_to_output) {
  int ret = -1;
  while (av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    // We want to make sure that we call av_packet_unref in every iteration
    auto packet = AVPacketScope(av_state_->packet_, av_packet_unref);

    if (packet->stream_index != av_state_->stream_id_) {
      continue;
    }

    ret = avcodec_send_packet(av_state_->codec_ctx_, packet.get());
    DALI_ENFORCE(
      ret >= 0,
      make_string("Failed to send packet to decoder: ", detail::av_error_string(ret)));

    ret = avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_);

    if (ret == AVERROR(EAGAIN)) {
      continue;
    }

    if (ret == AVERROR_EOF) {
      break;
    }

    LOG_LINE << "Read frame (ReadRegularFrame), index " << next_frame_idx_ << ", timestamp " <<
      std::setw(5)  << av_state_->frame_->pts << ", current copy " << copy_to_output << std::endl;
    if (!copy_to_output) {
      ++next_frame_idx_;
      return true;
    }

    CopyToOutput(data);
    ++next_frame_idx_;
    return true;
  }

  ret = avcodec_send_packet(av_state_->codec_ctx_, nullptr);
  DALI_ENFORCE(
    ret >= 0,
    make_string("Failed to send packet to decoder: ", detail::av_error_string(ret)));
  flush_state_ = true;

  return false;
}

void FramesDecoder::Reset() {
  next_frame_idx_ = 0;

  if (flush_state_) {
    flush_state_ = false;
  }

  int ret = av_seek_frame(av_state_->ctx_, av_state_->stream_id_, 0, AVSEEK_FLAG_FRAME);
  DALI_ENFORCE(
    ret >= 0,
    make_string(
      "Could not seek to the first frame of video \"",
      Filename(),
      "\", due to ",
      detail::av_error_string(ret)));
  if (av_state_->codec_) {
    avcodec_flush_buffers(av_state_->codec_ctx_);
  }
}

void FramesDecoder::SeekFrame(int frame_id) {
  // TODO(awolant): Optimize seeking:
  //  - when we seek next frame,
  //  - when we seek frame with the same keyframe as the current frame
  //  - for CFR, when we know pts, but don't know keyframes

  DALI_ENFORCE(
    frame_id >= 0 && frame_id < NumFrames(),
    make_string("Invalid seek frame id. frame_id = ", frame_id, ", num_frames = ", NumFrames()));

  auto &frame_entry = Index(frame_id);
  int keyframe_id = frame_entry.last_keyframe_id;
  auto &keyframe_entry = Index(keyframe_id);

  LOG_LINE << "Seeking to frame " << frame_id << " timestamp " << frame_entry.pts << std::endl;

  // Seeking clears av buffers, so reset flush state info
  if (flush_state_) {
    while (ReadFlushFrame(nullptr, false)) {}
    flush_state_ = false;
  }

  int ret = av_seek_frame(
    av_state_->ctx_, av_state_->stream_id_, keyframe_entry.pts, AVSEEK_FLAG_FRAME);
  DALI_ENFORCE(
    ret >= 0,
    make_string(
      "Failed to seek to frame ",
      frame_id,
      "with keyframe",
      keyframe_id,
      "in video \"",
      Filename(),
      "\" due to ",
      detail::av_error_string(ret)));

  avcodec_flush_buffers(av_state_->codec_ctx_);
  next_frame_idx_ = keyframe_id;

  for (int i = 0; i < (frame_id - keyframe_id); ++i) {
    ReadNextFrame(nullptr, false);
  }
}

bool FramesDecoder::ReadFlushFrame(uint8_t *data, bool copy_to_output) {
  if (avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_) < 0) {
    flush_state_ = false;
    return false;
  }

  if (copy_to_output) {
    CopyToOutput(data);
  }

  LOG_LINE << "Read frame (ReadFlushFrame), index " << next_frame_idx_ << " timestamp " <<
    std::setw(5) << av_state_->frame_->pts << ", current copy " << copy_to_output << std::endl;
  ++next_frame_idx_;

  // TODO(awolant): Figure out how to handle this during index building
  // Or when NumFrames in unavailible
  if (next_frame_idx_ >= NumFrames()) {
    next_frame_idx_ = -1;
  }

  return true;
}

bool FramesDecoder::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  if (!flush_state_) {
      if (ReadRegularFrame(data, copy_to_output)) {
        return true;
      }
  }
  return ReadFlushFrame(data, copy_to_output);
}

const IndexEntry &FramesDecoder::Index(int frame_id) const {
  if (!index_.has_value()) {
    DALI_FAIL("Functionality is unavailible when index is not built.");
  }

  return (*index_)[frame_id];
}
}  // namespace dali
