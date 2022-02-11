// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


namespace dali {

namespace detail {
std::string av_error_string(int ret) {
    static char msg[AV_ERROR_MAX_STRING_SIZE];
    memset(msg, 0, sizeof(msg));
    return std::string(av_make_error_string(msg, AV_ERROR_MAX_STRING_SIZE, ret));
}
}

using AVPacketScope = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;

void FramesDecoder::InitAvState() {
  av_state_->codec_ctx_ = avcodec_alloc_context3(av_state_->codec_);
  DALI_ENFORCE(av_state_->codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(av_state_->codec_ctx_, av_state_->codec_params_);
  DALI_ENFORCE(
    ret >= 0,
    make_string("Could not fill the codec based on parameters: ", detail::av_error_string(ret)));

  ret = avcodec_open2(av_state_->codec_ctx_, av_state_->codec_, nullptr);
  DALI_ENFORCE(
    ret == 0,
    make_string("Could not initialize codec context: ", detail::av_error_string(ret)));

  av_state_->frame_ = av_frame_alloc();
  DALI_ENFORCE(av_state_->frame_, "Could not allocate the av frame");

  av_state_->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state_->packet_, "Could not allocate av packet");
}

void FramesDecoder::FindVideoStream() {
  for (size_t i = 0; i < av_state_->ctx_->nb_streams; ++i) {
    av_state_->codec_params_ = av_state_->ctx_->streams[i]->codecpar;
    av_state_->codec_ = avcodec_find_decoder(av_state_->codec_params_->codec_id);

    if (av_state_->codec_ == nullptr) {
      continue;
    }

    if (av_state_->codec_->type == AVMEDIA_TYPE_VIDEO) {
        av_state_->stream_id_ = i;
        return;
    }
  }

  DALI_FAIL(make_string("Could not find a valid video stream in a file ", filename_));
}

FramesDecoder::FramesDecoder(const std::string &filename)
    : av_state_(std::make_unique<AvState>()), filename_(filename) {
  av_state_->ctx_ = avformat_alloc_context();
  DALI_ENFORCE(av_state_->ctx_, "Could not alloc avformat context");

  int ret = avformat_open_input(&av_state_->ctx_, filename.c_str(), nullptr, nullptr);
  DALI_ENFORCE(ret == 0, make_string("Failed to open video file at path ", filename, "due to ",
                                     detail::av_error_string(ret)));

  FindVideoStream();
  InitAvState();
  BuildIndex();
}

void FramesDecoder::BuildIndex() {
  // TODO(awolant): Optimize this function for:
  //  - CFR
  //  - index present in the header

  int last_keyframe = -1;
  while (ReadRegularFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = av_state_->frame_->key_frame;
    entry.pts = av_state_->frame_->pts;
    entry.is_flush_frame = false;

    if (entry.is_keyframe) {
      last_keyframe = index_.size();
    }
    entry.last_keyframe_id = last_keyframe;
    index_.push_back(entry);
  }
  while (ReadFlushFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = false;
    entry.pts = av_state_->frame_->pts;
    entry.is_flush_frame = true;
    entry.last_keyframe_id = last_keyframe;
    index_.push_back(entry);
  }
  Reset();
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
  int ret = av_seek_frame(av_state_->ctx_, av_state_->stream_id_, 0, AVSEEK_FLAG_FRAME);
  DALI_ENFORCE(
    ret >= 0,
    make_string(
      "Could not seek to the first frame of video ",
      filename_,
      "due to",
      detail::av_error_string(ret)));
  avcodec_flush_buffers(av_state_->codec_ctx_);
}

void FramesDecoder::SeekFrame(int frame_id) {
  // TODO(awolant): Optimize seeking:
  //  - when we seek next frame,
  //  - when we seek frame with the same keyframe as the current frame
  //  - for CFR, when we know pts, but don't know keyframes

  DALI_ENFORCE(
    frame_id >= 0 && frame_id < NumFrames(),
    make_string("Invalid seek frame id. frame_id = ", frame_id, ", num_frames = ", NumFrames()));

  auto &frame_entry = index_[frame_id];
  int keyframe_id = frame_entry.last_keyframe_id;
  auto &keyframe_entry = index_[keyframe_id];

  LOG_LINE << "Seeking to frame " << frame_id << " timestamp " << frame_entry.pts << std::endl;

  // Seeking clears av buffers, so reset flush state info
  if (flush_state_) {
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
      "in video ",
      filename_,
      "due to ",
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
}  // namespace dali
