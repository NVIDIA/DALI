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

#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"
#include <cuda.h>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/video/nvdecode/ColorSpace.h"
#include "dali/operators/reader/loader/video/nvdecode/NvCodecUtils.h"
#include <unistd.h>
#include "dali/core/error_handling.h"

namespace dali {
namespace detail {
int process_video_sequence(void *user_data, CUVIDEOFORMAT *video_format) {
  return video_format->min_num_decode_surfaces;
}

int process_picture_decode(void *user_data, CUVIDPICPARAMS *picture_params) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);
  frames_decoder->decode_success_ = false;

  CUDA_CALL(cuvidDecodePicture(frames_decoder->nvdecode_state_->decoder, picture_params));

  static int counter = 0;
  std::cout << "Call " << counter << ", pic index " << picture_params->CurrPicIdx << std::endl;
  ++counter;


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

  CUVIDGETDECODESTATUS DecodeStatus;
  memset(&DecodeStatus, 0, sizeof(DecodeStatus));
  CUresult result = cuvidGetDecodeStatus(frames_decoder->nvdecode_state_->decoder, picture_params->CurrPicIdx, &DecodeStatus);
  if (result != CUDA_SUCCESS) {
    CUDA_CALL(cuvidUnmapVideoFrame(frames_decoder->nvdecode_state_->decoder, frame));

    if (DecodeStatus.decodeStatus == cuvidDecodeStatus_Invalid) {
      return 0;
    }
  }

  if (frames_decoder->current_copy_to_output_) {
    CUDA_CALL(cudaDeviceSynchronize());
    Nv12ToColor32(
      (uint8_t *)frame,
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
}

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename) :
  FramesDecoder(filename) {
    // CUcontext context;
    // CUDA_CALL(cuCtxGetCurrent(&context));

    // decoder_ = std::make_unique<NvDecoder>(context, true, cudaVideoCodec_MPEG4);

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
  // DALI_ENFORCE(
  //   frame_id >= 0 && frame_id < NumFrames(),
  //   make_string("Invalid seek frame id. frame_id = ", frame_id, ", num_frames = ", NumFrames()));

  // Reset();
  // CUcontext context;
  // CUDA_CALL(cuCtxGetCurrent(&context));
  // decoder_ = std::make_unique<NvDecoder>(context, true, cudaVideoCodec_MPEG4);
  // // decoder_->setReconfigParams(nullptr, nullptr);

  // for (int i = 0; i < frame_id; ++i) {
  //   ReadNextFrame(nullptr, false);
  // }

  // CUcontext context;
  // CUDA_CALL(cuCtxGetCurrent(&context));
  // decoder_ = std::make_unique<NvDecoder>(context, true, cudaVideoCodec_MPEG4);
  FramesDecoder::SeekFrame(frame_id);
}

bool FramesDecoderGpu::DecodeFrame(uint8_t *data, bool copy_to_output) {
  // return FramesDecoder::DecodeFrame(data, true);
  decode_success_ = false;
  current_frame_output_ = data;
  current_copy_to_output_ = copy_to_output;

  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = av_state_->packet_->data;
  packet->payload_size = av_state_->packet_->size;
  packet->flags = CUVID_PKT_TIMESTAMP;
  packet->timestamp = index_[current_frame_-1].pts;

  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

  // int n_frames_returned = decoder_->Decode(av_state_->packet_->data, av_state_->packet_->size, CUVID_PKT_TIMESTAMP, index_[current_frame_-1].pts);


  
  // if (copy_to_output)
  //   cout << "GPU: "  << av_state_->packet_->size << endl;

  // if (n_frames_returned == 0) {
  //   // n_frames_returned = decoder_->Decode(av_state_->packet_->data, av_state_->packet_->size, CUVID_PKT_TIMESTAMP, index_[current_frame_-1].pts);
  //   if (n_frames_returned == 0) {
  //     return false;
  // }
  // }

  // if (n_frames_returned > 1) {
  //   DALI_FAIL("dużo ramek");
  // }

  // if (!copy_to_output) {
  //   return true;
  // }
  
  // int64_t timestamp = 0;
  // uint8_t *frame = decoder_->GetFrame(&timestamp);
  // CUDA_CALL(cudaDeviceSynchronize());
  // Nv12ToColor32(frame, Width(), data, Width()* 3, Width(), Height(), 1);
  // CUDA_CALL(cudaDeviceSynchronize());


  return decode_success_;
}


}  // namespace dali
