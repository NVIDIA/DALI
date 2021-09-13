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

#include "dali/operators/reader/loader/video_loader_cpu.h"

namespace dali {

VideoFileCPU::VideoFileCPU(std::string &filename) {
  ctx_ = avformat_alloc_context();
  avformat_open_input(&ctx_, filename.c_str(), nullptr, nullptr);

  for (size_t i = 0; i < ctx_->nb_streams; ++i) {
    auto stream = ctx_->streams[i];
    codec_params_ = ctx_->streams[i]->codecpar;
    codec_ = avcodec_find_decoder(codec_params_->codec_id);

    if (codec_->type == AVMEDIA_TYPE_VIDEO) {
        stream_id_ = i;
        break;
    }
  }

  codec_ctx_ = avcodec_alloc_context3(codec_);
  avcodec_parameters_to_context(codec_ctx_, codec_params_);
  avcodec_open2(codec_ctx_, codec_, nullptr);
  frame_ = av_frame_alloc();
  packet_ = av_packet_alloc();

  BuildIndex();
}

void VideoFileCPU::BuildIndex() {
  int last_keyframe = -1;
  while (ReadRegularFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = frame_->key_frame;
    entry.pts = frame_->pts;
    entry.is_flush_frame = false;
    
    if (entry.is_keyframe) {
      last_keyframe = index_.size();
    }
    entry.last_keyframe_id = last_keyframe;
    index_.push_back(entry);
    ++num_frames_;
  }
  while(ReadFlushFrame(nullptr, false)) {
    IndexEntry entry;
    entry.is_keyframe = false;
    entry.pts = frame_->pts;
    entry.is_flush_frame = true;
    entry.last_keyframe_id = last_keyframe;
    index_.push_back(entry);
    ++num_frames_;
  }
  Reset();
}

void VideoFileCPU::CopyToOutput(uint8_t *data) {
  dest_[0] = data;
  dest_linesize_[0] = frame_->width * 3;
  if (sws_scale(sws_ctx_, frame_->data, frame_->linesize, 0, frame_->height, dest_, dest_linesize_) < 0) {
    DALI_FAIL("");
  }
}

bool VideoFileCPU::ReadRegularFrame(uint8_t *data, bool copy_to_output) {
  while(av_read_frame(ctx_, packet_) >= 0) {
    if (packet_->stream_index != stream_id_) {
      continue;
    }

    if (avcodec_send_packet(codec_ctx_, packet_) < 0) {
      DALI_FAIL("Failed to send packet to decoder");
    }

    int ret = avcodec_receive_frame(codec_ctx_, frame_);

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      continue;
    }

    if (!copy_to_output) {
      return true;
    }

    if (sws_ctx_ == nullptr) {
      sws_ctx_ = sws_getContext(
        Width(), 
        Height(), 
        codec_ctx_->pix_fmt, 
        Width(), 
        Height(), 
        AV_PIX_FMT_RGB24, 
        SWS_BILINEAR, 
        nullptr, 
        nullptr, 
        nullptr);
    }

    CopyToOutput(data);
    return true;
  }

  return false;
}

void VideoFileCPU::Reset() {
  av_seek_frame(ctx_, stream_id_, 0, AVSEEK_FLAG_FRAME);
  avcodec_flush_buffers(codec_ctx_);
}

void VideoFileCPU::SeekFrame(int frame_id) {
  auto &frame_entry = index_[frame_id];
  int keyframe_id = frame_entry.last_keyframe_id;
  auto &keyframe_entry = index_[keyframe_id];

  // Seeking while on flush frame. Need to reset flush state
  if (flush_state_) {
    flush_state_ = false;
  }
 
  av_seek_frame(ctx_, stream_id_, keyframe_entry.pts, AVSEEK_FLAG_FRAME);
  avcodec_flush_buffers(codec_ctx_);

  for (int i = 0; i < (frame_id - keyframe_id); ++i) {
    ReadNextFrame(nullptr, false);
  }
}

bool VideoFileCPU::ReadFlushFrame(uint8_t *data, bool copy_to_output) {
  if (!flush_state_) {
    avcodec_send_packet(codec_ctx_, nullptr);
    flush_state_ = true;
  }

  if (avcodec_receive_frame(codec_ctx_, frame_) < 0) {
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


void VideoLoaderCPU::ReadSample(Tensor<CPUBackend> &sample) {
  auto &sample_span = sample_spans_[current_index_];
  auto &video_file = video_files_[0];

  ++current_index_;
  MoveToNextShard(current_index_);

  sample.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
  sample.Resize(
    TensorShape<4>{sequence_len_, video_file.Width(), video_file.Height(), video_file.channels_});

  auto data = sample.mutable_data<uint8_t>();

  for (int i = 0; i < sequence_len_; ++i) {
    video_file.SeekFrame(sample_span.start_ + i * sample_span.stride_);     //This seek can be optimized - for consecutive frames not needed etc.
    video_file.ReadNextFrame(data + i * video_file.FrameSize());
  }
}

Index VideoLoaderCPU::SizeImpl() {
  return sample_spans_.size();
}

void VideoLoaderCPU::PrepareMetadataImpl() {
  for (auto &filename : filenames_) {
    video_files_.push_back(VideoFileCPU(filename));
  }

  for (int start = 0; start + stride_ * sequence_len_ <= video_files_[0].NumFrames(); start += stride_ * sequence_len_) {
    sample_spans_.push_back(VideoSampleSpan(start, start + stride_ * sequence_len_, stride_));
  }
}

void VideoLoaderCPU::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
}

}  // namespace dali
