// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/frames_decoder_cpu.h"
#include <algorithm>
#include <string>
#include "dali/core/error_handling.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/transpose/transpose.h"
#include <cstring>
#include <iomanip>
#include "dali/operators/video/video_utils.h"

namespace dali {

FramesDecoderCpu::FramesDecoderCpu(const std::string &filename, DALIImageType image_type)
    : FramesDecoderBase(filename, image_type) {
  is_valid_ = is_valid_ && SelectVideoStream();
}

FramesDecoderCpu::FramesDecoderCpu(const char *memory_file, size_t memory_file_size, std::string_view source_info, DALIImageType image_type)
  : FramesDecoderBase(memory_file, memory_file_size, source_info, image_type) {
  is_valid_ = is_valid_ && SelectVideoStream();
}

void FramesDecoderCpu::CopyFrame(uint8_t *dst, const uint8_t *src) {
  std::memcpy(dst, src, FrameSize());
}

void FramesDecoderCpu::Flush() {
  LOG_LINE << "FramesDecoderCpu::Flush" << std::endl;
  avcodec_flush_buffers(codec_ctx_);
  if (flush_state_) {
    LOG_LINE << "Flushing frames" << std::endl;
    while (ReadFlushFrame(nullptr)) {}
    flush_state_ = false;
  }
}

bool FramesDecoderCpu::ReadNextFrame(uint8_t *data) {
  LOG_LINE << "FramesDecoderCpu::ReadNextFrame: next_frame_idx_=" << next_frame_idx_ << std::endl;
  // No more frames in the file
  if (next_frame_idx_ == -1) {
    return false;
  }

  if (!flush_state_) {
    if (ReadRegularFrame(data)) {
      return true;
    }
  }
  return ReadFlushFrame(data);
}


void PlanarToInterleaved(uint8_t *output, const uint8_t *input, int64_t height, int64_t width, int64_t channels) {
  std::array<int, 3> perm = {1, 2, 0};
  TensorView<StorageCPU, uint8_t> output_view{output, {height, width, channels}};
  TensorView<StorageCPU, const uint8_t> input_view{input, {channels, height, width}};
  kernels::TransposeGrouped(output_view, input_view, make_cspan(perm));
}

void FramesDecoderCpu::CopyToOutput(uint8_t *data) {
  uint8_t *sws_output_data = data;
  AVPixelFormat sws_output_format = AV_PIX_FMT_RGB24;
  if (image_type_ == DALI_YCbCr || Width() % 32) {
    // in some cases, when Width() % 32, sws_scale go past the provided memory for
    // conversion output, so we need the memory allocated with extra padding
    auto extra_padding = 32;
    tmp_buffer_.resize(FrameSize() + extra_padding);
    sws_output_data = tmp_buffer_.data();
    if (image_type_ == DALI_YCbCr) {
      sws_output_format = AV_PIX_FMT_YUV444P;
    }
  }
  if (!sws_ctx_) {
    sws_ctx_ = std::unique_ptr<SwsContext, decltype(&sws_freeContext)>(
      sws_getContext(
        Width(),
        Height(),
        codec_ctx_->pix_fmt,
        Width(),
        Height(),
        sws_output_format,
        SWS_BILINEAR|SWS_FULL_CHR_H_INT|SWS_ACCURATE_RND,
        nullptr,
        nullptr,
        nullptr),
      sws_freeContext);
    DALI_ENFORCE(sws_ctx_, "Could not create sw context");
  }

  uint8_t *dest[4] = {sws_output_data, nullptr, nullptr, nullptr};
  int dest_linesize[4] = {frame_->width * Channels(), 0, 0, 0};

  LOG_LINE << "Converting frame data to format " << (sws_output_format == AV_PIX_FMT_RGB24 ? "RGB" : "YUV") << std::endl;
  int ret = sws_scale(
    sws_ctx_.get(),
    frame_->data,
    frame_->linesize,
    0,
    frame_->height,
    dest,
    dest_linesize);

  if (image_type_ == DALI_YCbCr) {
    LOG_LINE << "Converting planar YUV to interleaved" << std::endl;
    PlanarToInterleaved(data, sws_output_data, Height(), Width(), Channels());
  } else if (Width() % 32) {
    std::copy(sws_output_data, sws_output_data + FrameSize(), data);
  }

  DALI_ENFORCE(ret >= 0,
               make_string("Could not convert frame data: ", av_error_string(ret)));
}

bool FramesDecoderCpu::ReadRegularFrame(uint8_t *data) {
  int ret = -1;
  bool copy_to_output = data != nullptr;
  while (true) {
    ret = av_read_frame(ctx_, packet_);
    auto packet = AVPacketScope(packet_, av_packet_unref);
    if (ret != 0) {
      break;  // End of file
    }

    if (packet->stream_index != stream_id_) {
      continue;
    }
    ret = avcodec_send_packet(codec_ctx_, packet.get());
    DALI_ENFORCE(ret >= 0,
                 make_string("Failed to send packet to decoder: ", av_error_string(ret)));

    ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret == AVERROR(EAGAIN)) {
      continue;
    }

    if (ret == AVERROR_EOF) {
      break;
    }

    LOG_LINE << (copy_to_output ? "Read" : "Skip") << " frame (ReadRegularFrame), index "
             << next_frame_idx_ << ", timestamp " << std::setw(5) << frame_->pts
             << std::endl;
    if (!copy_to_output) {
      ++next_frame_idx_;
      return true;
    }

    CopyToOutput(data);
    ++next_frame_idx_;
    return true;
  }

  ret = avcodec_send_packet(codec_ctx_, nullptr);
  DALI_ENFORCE(ret >= 0,
               make_string("Failed to send packet to decoder: ", av_error_string(ret)));
  flush_state_ = true;

  return false;
}

bool FramesDecoderCpu::ReadFlushFrame(uint8_t *data) {
  bool copy_to_output = data != nullptr;
  if (avcodec_receive_frame(codec_ctx_, frame_) < 0) {
    flush_state_ = false;
    return false;
  }

  if (copy_to_output) {
    CopyToOutput(data);
  }

  LOG_LINE << (copy_to_output ? "Read" : "Skip") << "frame (ReadFlushFrame), index "
           << next_frame_idx_ << " timestamp " << std::setw(5) << frame_->pts
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

void FramesDecoderCpu::Reset() {
  LOG_LINE << "Resetting decoder" << std::endl;
  FramesDecoderBase::Reset();
  flush_state_ = false;
}

bool FramesDecoderCpu::SelectVideoStream(int stream_id) {
  if (!FramesDecoderBase::SelectVideoStream(stream_id)) {
    return false;
  }

  assert(stream_id_ >= 0 && stream_id_ < static_cast<int>(ctx_->nb_streams));
  assert(codec_params_);
  AVCodecID codec_id = codec_params_->codec_id;

  static constexpr std::array<AVCodecID, 7> codecs = {
    AVCodecID::AV_CODEC_ID_H264,
    AVCodecID::AV_CODEC_ID_HEVC,
    AVCodecID::AV_CODEC_ID_VP8,
    AVCodecID::AV_CODEC_ID_VP9,
    AVCodecID::AV_CODEC_ID_MJPEG,
    // Those are not supported by our compiled version of libavcodec,
    // AVCodecID::AV_CODEC_ID_AV1,
    // AVCodecID::AV_CODEC_ID_MPEG4,
  };

  if (std::find(codecs.begin(), codecs.end(), codec_id) == codecs.end()) {
    DALI_WARN(make_string("Codec ", codec_id, " (", avcodec_get_name(codec_id),
                          ") is not supported by the CPU variant of this operator."));
    return false;
  }

  if ((codec_ = avcodec_find_decoder(codec_params_->codec_id)) == nullptr) {
    LOG_LINE << "No decoder found for codec " << avcodec_get_name(codec_params_->codec_id)
             << " (codec_id=" << codec_params_->codec_id << ")" << std::endl;
    return false;
  }

  codec_ctx_.reset(avcodec_alloc_context3(codec_));
  DALI_ENFORCE(codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(codec_ctx_, codec_params_);
  DALI_ENFORCE(ret >= 0, make_string("Could not fill the codec based on parameters: ",
                                     av_error_string(ret)));

  ret = avcodec_open2(codec_ctx_, codec_, nullptr);
  if (ret != 0) {
    DALI_WARN(make_string("Could not initialize codec context: ", av_error_string(ret)));
    return false;
  }

  frame_.reset(av_frame_alloc());
  if (!frame_) {
    DALI_WARN("Could not allocate the av frame");
    return false;
  }

  return true;
}

}  // namespace dali
