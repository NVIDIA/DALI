// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

#include <cuda.h>
#include <unistd.h>

#include <string>
#include <memory>

#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/video/nvdecode/ColorSpace.h"

namespace dali {
namespace detail {
int process_video_sequence(void *user_data, CUVIDEOFORMAT *video_format) {
  return video_format->min_num_decode_surfaces;
}

int process_picture_decode(void *user_data, CUVIDPICPARAMS *picture_params) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);
  if (frames_decoder->flush_) {
    frames_decoder->flush_ = false;
    frames_decoder->decode_success_ = false;
    return 0;
  }

  frames_decoder->decode_success_ = false;

  CUDA_CALL(cuvidDecodePicture(frames_decoder->nvdecode_state_->decoder, picture_params));

  // Copy decoded frame to output
  CUVIDPROCPARAMS videoProcessingParameters = {};
  videoProcessingParameters.progressive_frame = !picture_params->field_pic_flag;
  videoProcessingParameters.second_field = 1;
  videoProcessingParameters.top_field_first = picture_params->bottom_field_flag ^ 1;
  videoProcessingParameters.unpaired_field = 0;
  videoProcessingParameters.output_stream = 0;

  CUdeviceptr frame = 0;
  unsigned int pitch = 0;

  CUDA_CALL(cuvidMapVideoFrame(
    frames_decoder->nvdecode_state_->decoder,
    picture_params->CurrPicIdx,
    &frame,
    &pitch,
    &videoProcessingParameters));

  if (frames_decoder->current_copy_to_output_) {
    // CUDA_CALL(cudaDeviceSynchronize());
    Nv12ToColor32(
      reinterpret_cast<uint8_t *>(frame),
      pitch,
      frames_decoder->current_frame_output_,
      frames_decoder->Width()* 3,
      frames_decoder->Width(),
      frames_decoder->Height(),
      1);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  CUDA_CALL(cuStreamSynchronize(0));
  CUDA_CALL(cuvidUnmapVideoFrame(frames_decoder->nvdecode_state_->decoder, frame));

  frames_decoder->decode_success_ = true;
  return 1;
}
}  // namespace detail

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename) :
  // FramesDecoder(filename), frame_buffer_(num_decode_surfaces_) {
    FramesDecoder(filename), frame_buffer_(60) {
    nvdecode_state_ = std::make_unique<NvDecodeState>();

    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    DALI_ENFORCE(av_bsf_alloc(bsf, &bsfc_) >= 0);
    DALI_ENFORCE(avcodec_parameters_copy(
      bsfc_->par_in, av_state_->ctx_->streams[0]->codecpar) >= 0);
    DALI_ENFORCE(av_bsf_init(bsfc_) >= 0);

    filtered_packet_ = av_packet_alloc();
    DALI_ENFORCE(filtered_packet_, "Could not allocate av packet");

    // Create nv decoder
    CUVIDDECODECREATEINFO decoder_info = {};
    memset(&decoder_info, 0, sizeof(CUVIDDECODECREATEINFO));

    decoder_info.bitDepthMinus8 = 0;
    decoder_info.ChromaFormat = cudaVideoChromaFormat_420;
    decoder_info.CodecType = cudaVideoCodec_H264;
    decoder_info.ulHeight = Height();
    decoder_info.ulWidth = Width();
    decoder_info.ulMaxHeight = Height();
    decoder_info.ulMaxWidth = Width();
    decoder_info.ulTargetHeight = Height();
    decoder_info.ulTargetWidth = Width();
    decoder_info.ulNumDecodeSurfaces = num_decode_surfaces_;
    decoder_info.ulNumOutputSurfaces = 2;

    CUDA_CALL(cuvidCreateDecoder(&nvdecode_state_->decoder, &decoder_info));

    // Create nv parser
    CUVIDPARSERPARAMS parser_info;
    memset(&parser_info, 0, sizeof(CUVIDPARSERPARAMS));
    parser_info.CodecType = cudaVideoCodec_H264;
    parser_info.ulMaxNumDecodeSurfaces = 1;
    parser_info.ulMaxDisplayDelay = 0;
    parser_info.pUserData = this;
    parser_info.pfnSequenceCallback = detail::process_video_sequence;
    parser_info.pfnDecodePicture = detail::process_picture_decode;
    parser_info.pfnDisplayPicture = nullptr;

    CUDA_CALL(cuvidCreateVideoParser(&nvdecode_state_->parser, &parser_info));

    // Init internal frame buffer
    for (int i = 0; i < frame_buffer_.size(); ++i) {
      frame_buffer_[i].frame_.resize(FrameSize());
      frame_buffer_[i].pts_ = -1;
    }

    current_frame_buffer_.resize(FrameSize());
}

void FramesDecoderGpu::SeekFrame(int frame_id) {
  flush_ = true;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));
  flush_ = false;
  last_frame_read_ = false;

  for (int i = 0; i < frame_buffer_.size(); ++i) {
    frame_buffer_[i].pts_ = -1;
  }

  while (piped_pts_.size() > 0) {
    piped_pts_.pop();
  }

  FramesDecoder::SeekFrame(frame_id);
}

bool FramesDecoderGpu::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  if (current_frame_ == -1) {
    return false;
  }
  // Maybe requested frame is already in the buffer?
  if (frame_buffer_[current_frame_].pts_ != -1) {
    if (copy_to_output) {
      copyD2D(data, frame_buffer_[current_frame_].frame_.data(), FrameSize());
    }
    frame_buffer_[current_frame_].pts_ = -1;

    ++current_frame_;
    return true;
  }

  decode_success_ = false;
  current_copy_to_output_ = copy_to_output;
  current_frame_output_ = current_frame_buffer_.data();

  while (av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    if (av_state_->packet_->stream_index != av_state_->stream_id_) {
      continue;
    }

    piped_pts_.push(av_state_->packet_->pts);

    if (filtered_packet_->data) {
      av_packet_unref(filtered_packet_);
    }

    DALI_ENFORCE(av_bsf_send_packet(bsfc_, av_state_->packet_) >= 0);
    DALI_ENFORCE(av_bsf_receive_packet(bsfc_, filtered_packet_) >= 0);

    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = filtered_packet_->data;
    packet->payload_size = filtered_packet_->size;
    packet->flags = CUVID_PKT_TIMESTAMP;
    packet->timestamp = av_state_->packet_->pts;

    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

    if (decode_success_) {
      int current_pts = piped_pts_.front();
      piped_pts_.pop();

      int requested_pts = index_[current_frame_].pts;

      if (current_pts == requested_pts) {
        // Currently returned frame is actually the one we wanted
        if (copy_to_output) {
          copyD2D(data, current_frame_buffer_.data(), FrameSize());
        }
        ++current_frame_;
        return true;
      } else {
        int found_frame_index = 0;
        while (current_pts != index_[found_frame_index].pts) {
          ++found_frame_index;
        }

        frame_buffer_[found_frame_index].pts_ = current_pts;
        copyD2D(
          frame_buffer_[found_frame_index].frame_.data(),
          current_frame_buffer_.data(),
          FrameSize());
      }
    }
  }

  if (!last_frame_read_) {
    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = nullptr;
    packet->payload_size = 0;
    packet->flags = CUVID_PKT_ENDOFSTREAM;
    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

    last_frame_read_ = true;
    piped_pts_.pop();
    if (copy_to_output) {
      copyD2D(data, current_frame_buffer_.data(), FrameSize());
    }
    current_frame_ = -1;
    return true;
  }

  return false;
}

void FramesDecoderGpu::Reset() {
  flush_ = true;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));
  flush_ = false;
  last_frame_read_ = false;

  for (int i = 0; i < frame_buffer_.size(); ++i) {
    frame_buffer_[i].pts_ = -1;
  }

  while (piped_pts_.size() > 0) {
    piped_pts_.pop();
  }

  FramesDecoder::Reset();
}
}  // namespace dali
