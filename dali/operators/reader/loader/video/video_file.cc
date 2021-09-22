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

#include "dali/operators/reader/loader/video/video_file.h"
#include "dali/core/error_handling.h"

namespace dali {

void VideoFileCPU::InitAvState() {
  av_state_->codec_ctx_ = avcodec_alloc_context3(av_state_->codec_);
  DALI_ENFORCE(av_state_->codec_ctx_, "Could not create av codec context");

  DALI_ENFORCE(
    avcodec_parameters_to_context(av_state_->codec_ctx_, av_state_->codec_params_) >= 0,
    "Could not fill the codec based on parameters");

  DALI_ENFORCE(
    avcodec_open2(av_state_->codec_ctx_, av_state_->codec_, nullptr) == 0,
    "Could not initialize codec context");

  av_state_->frame_ = av_frame_alloc();
  DALI_ENFORCE(av_state_->frame_, "Could not allocate the av frame");

  av_state_->packet_ = av_packet_alloc();
  DALI_ENFORCE(av_state_->packet_, "Could not allocate av packet");
}

void VideoFileCPU::FindVideoStream() {
  for (size_t i = 0; i < av_state_->ctx_->nb_streams; ++i) {
    auto stream = av_state_->ctx_->streams[i];
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

  DALI_FAIL(make_string("Could not find a valid video stream in file ", filename_));
}

VideoFileCPU::VideoFileCPU(const std::string &filename) : 
  av_state_(std::make_unique<AvState>()),
  filename_(filename) {
  av_state_->ctx_ = avformat_alloc_context();
  DALI_ENFORCE(av_state_->ctx_, "Could not create avformat context");

  DALI_ENFORCE(
    avformat_open_input(&av_state_->ctx_, filename.c_str(), nullptr, nullptr) == 0,
    make_string("Failed to open video file at path ", filename));

  FindVideoStream();
  InitAvState();
  BuildIndex();
}

void VideoFileCPU::BuildIndex() {
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
  while(ReadFlushFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = false;
    entry.pts = av_state_->frame_->pts;
    entry.is_flush_frame = true;
    entry.last_keyframe_id = last_keyframe;
    index_.push_back(entry);
  }
  Reset();
}

void VideoFileCPU::CopyToOutput(uint8_t *data) {
  dest_[0] = data;
  dest_linesize_[0] = av_state_->frame_->width * Channels();
  if (sws_scale(av_state_->sws_ctx_, av_state_->frame_->data, av_state_->frame_->linesize, 0, av_state_->frame_->height, dest_, dest_linesize_) < 0) {
    DALI_FAIL("Could not convert frame data to RGB");
  }
}

bool VideoFileCPU::ReadRegularFrame(uint8_t *data, bool copy_to_output) {
  while(av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    if (av_state_->packet_->stream_index != av_state_->stream_id_) {
      continue;
    }

    if (avcodec_send_packet(av_state_->codec_ctx_, av_state_->packet_) < 0) {
      DALI_FAIL("Failed to send packet to decoder");
    }

    int ret = avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_);

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      continue;
    }

    if (!copy_to_output) {
      return true;
    }

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

    CopyToOutput(data);
    return true;
  }

  avcodec_send_packet(av_state_->codec_ctx_, nullptr);
  flush_state_ = true;

  return false;
}

void VideoFileCPU::Reset() {
  DALI_ENFORCE(
    av_seek_frame(av_state_->ctx_, av_state_->stream_id_, 0, AVSEEK_FLAG_FRAME) >= 0,
    make_string("Could not seek to the first frame of video ", filename_));
  avcodec_flush_buffers(av_state_->codec_ctx_);
}

void VideoFileCPU::SeekFrame(int frame_id) {
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

  // Seeking while on flush frame. Need to reset flush state
  if (flush_state_) {
    flush_state_ = false;
  }
 
  DALI_ENFORCE(
    av_seek_frame(av_state_->ctx_, av_state_->stream_id_, keyframe_entry.pts, AVSEEK_FLAG_FRAME) >= 0,
    make_string("Failed to seek to frame ", frame_id, "with keyframe", keyframe_id, "in video ", filename_));

  avcodec_flush_buffers(av_state_->codec_ctx_);

  for (int i = 0; i < (frame_id - keyframe_id); ++i) {
    ReadNextFrame(nullptr, false);
  }
}

bool VideoFileCPU::ReadFlushFrame(uint8_t *data, bool copy_to_output) {
  if (avcodec_receive_frame(av_state_->codec_ctx_, av_state_->frame_) < 0) {
    flush_state_ = false;
    return false;
  }

  if (copy_to_output) {
    CopyToOutput(data);
  }

  return true;
}

void VideoFileCPU::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  if (!flush_state_) {
      if (ReadRegularFrame(data, copy_to_output)) {
        return;
      }
  }
  if (ReadFlushFrame(data, copy_to_output)) {
    return;
  }

  Reset();

  if (!ReadRegularFrame(data, copy_to_output)) {
    DALI_FAIL("Error while reading frame");
  }
}
}  // namespace dali
