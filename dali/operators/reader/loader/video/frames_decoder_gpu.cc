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
    CUDA_CALL(cudaDeviceSynchronize());
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
  FramesDecoder(filename) {
    nvdecode_state_ = std::make_unique<NvDecodeState>();

    // Create nv decoder
    CUVIDDECODECREATEINFO decoder_info = {};
    memset(&decoder_info, 0, sizeof(CUVIDDECODECREATEINFO));

    decoder_info.bitDepthMinus8 = 0;
    decoder_info.ChromaFormat = cudaVideoChromaFormat_420;
    decoder_info.CodecType = cudaVideoCodec_MPEG4;
    decoder_info.ulHeight = Height();
    decoder_info.ulWidth = Width();
    decoder_info.ulMaxHeight = Height();
    decoder_info.ulMaxWidth = Width();
    decoder_info.ulTargetHeight = Height();
    decoder_info.ulTargetWidth = Width();
    decoder_info.ulNumDecodeSurfaces = 4;
    decoder_info.ulNumOutputSurfaces = 1;

    CUDA_CALL(cuvidCreateDecoder(&nvdecode_state_->decoder, &decoder_info));

    // Create nv parser
    CUVIDPARSERPARAMS parser_info;
    memset(&parser_info, 0, sizeof(CUVIDPARSERPARAMS));
    parser_info.CodecType = cudaVideoCodec_MPEG4;
    parser_info.ulMaxNumDecodeSurfaces = 1;
    parser_info.ulMaxDisplayDelay = 0;
    parser_info.pUserData = this;
    parser_info.pfnSequenceCallback = detail::process_video_sequence;
    parser_info.pfnDecodePicture = detail::process_picture_decode;
    parser_info.pfnDisplayPicture = nullptr;

    CUDA_CALL(cuvidCreateVideoParser(&nvdecode_state_->parser, &parser_info));
}

void FramesDecoderGpu::SeekFrame(int frame_id) {
  flush_ = true;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

  last_frame_read_ = false;

  FramesDecoder::SeekFrame(frame_id);
}

bool FramesDecoderGpu::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  decode_success_ = false;
  current_copy_to_output_ = copy_to_output;
  current_frame_output_ = data;

  while (av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    if (av_state_->packet_->stream_index != av_state_->stream_id_) {
      continue;
    }

    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = av_state_->packet_->data;
    packet->payload_size = av_state_->packet_->size;
    packet->flags = CUVID_PKT_TIMESTAMP;
    packet->timestamp = av_state_->packet_->pts;

    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

    if (decode_success_) {
      return true;
    }
  }

  if (!last_frame_read_) {
    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = av_state_->packet_->data;
    packet->payload_size = av_state_->packet_->size;
    packet->flags = CUVID_PKT_TIMESTAMP;
    packet->timestamp = av_state_->packet_->pts;

    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

    last_frame_read_ = true;
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

  last_frame_read_ = false;

  FramesDecoder::Reset();
}
}  // namespace dali
