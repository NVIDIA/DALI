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
#include <iomanip>

#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/video/nvdecode/color_space.h"

namespace dali {
namespace detail {
int process_video_sequence(void *user_data, CUVIDEOFORMAT *video_format) {
  return video_format->min_num_decode_surfaces;
}

int process_picture_decode(void *user_data, CUVIDPICPARAMS *picture_params) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);

  return frames_decoder->ProcessPictureDecode(user_data, picture_params);
}
}  // namespace detail

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename, cudaStream_t stream) :
    FramesDecoder(filename),
    frame_buffer_(num_decode_surfaces_),
    stream_(stream) {
    nvdecode_state_ = std::make_unique<NvDecodeState>();

    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    DALI_ENFORCE(av_bsf_alloc(bsf, &bsfc_) >= 0);
    DALI_ENFORCE(avcodec_parameters_copy(
      bsfc_->par_in, av_state_->ctx_->streams[0]->codecpar) >= 0);
    DALI_ENFORCE(av_bsf_init(bsfc_) >= 0);

    filtered_packet_ = av_packet_alloc();
    DALI_ENFORCE(filtered_packet_, "Could not allocate av packet");

    // Create nv decoder
    CUVIDDECODECREATEINFO decoder_info;
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
    parser_info.ulMaxNumDecodeSurfaces = num_decode_surfaces_;
    parser_info.ulMaxDisplayDelay = 0;
    parser_info.pUserData = this;
    parser_info.pfnSequenceCallback = detail::process_video_sequence;
    parser_info.pfnDecodePicture = detail::process_picture_decode;
    parser_info.pfnDisplayPicture = nullptr;

    CUDA_CALL(cuvidCreateVideoParser(&nvdecode_state_->parser, &parser_info));

    // Init internal frame buffer
    // TODO(awolant): Check, if continous buffer would be faster
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
      frame_buffer_[i].frame_.resize(FrameSize());
      frame_buffer_[i].pts_ = -1;
    }
}

int FramesDecoderGpu::ProcessPictureDecode(void *user_data, CUVIDPICPARAMS *picture_params) {
  // Sending empty packet will call this callback.
  // If we want to flush the decoder, we do not need to do anything here
  if (flush_) {
    return 0;
  }

  CUDA_CALL(cuvidDecodePicture(nvdecode_state_->decoder, picture_params));

  // Process decoded frame for output
  CUVIDPROCPARAMS videoProcessingParameters = {};
  videoProcessingParameters.progressive_frame = !picture_params->field_pic_flag;
  videoProcessingParameters.second_field = 1;
  videoProcessingParameters.top_field_first = picture_params->bottom_field_flag ^ 1;
  videoProcessingParameters.unpaired_field = 0;
  videoProcessingParameters.output_stream = stream_;

  uint8_t *frame_output = nullptr;

  // Take pts of the currently decoded frame
  int current_pts = piped_pts_.front();
  piped_pts_.pop();

  // current_pts is pts of frame that came from the decoder
  // NextFramePts() is pts of the frame that we want to return
  // in this call to ReadNextFrame
  // If they are the same, we just return this frame
  // If not, we store it in the buffer for later

  if (current_pts == NextFramePts()) {
    // Currently decoded frame is actually the one we wanted
    frame_returned_ = true;

    LOG_LINE << "Read frame, index " << next_frame_idx_ << ", timestamp " <<
        std::setw(5) << current_pts << ", current copy " << current_copy_to_output_ << std::endl;

    if (current_copy_to_output_ == false) {
      return 1;
    }
    frame_output = current_frame_output_;
  } else {
    // Put currently decoded frame to the buffer for later
    auto &slot = FindEmptySlot();
    slot.pts_ = current_pts;
    frame_output = slot.frame_.data();
  }

  CUdeviceptr frame = {};
  unsigned int pitch = 0;

  CUDA_CALL(cuvidMapVideoFrame(
    nvdecode_state_->decoder,
    picture_params->CurrPicIdx,
    &frame,
    &pitch,
    &videoProcessingParameters));

  // TODO(awolant): Benchmark, if copy would be faster
  yuv_to_rgb(
    reinterpret_cast<uint8_t *>(frame),
    pitch,
    frame_output,
    Width()* 3,
    Width(),
    Height(),
    stream_);
  // TODO(awolant): Alterantive is to copy the data to a buffer
  // and then process it on the stream. Check, if this is faster, when
  // the benchmark is ready.
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cuvidUnmapVideoFrame(nvdecode_state_->decoder, frame));

  return 1;
}

void FramesDecoderGpu::SeekFrame(int frame_id) {
  // TODO(awolant): This seek can be optimized - for consecutive frames not needed etc.
  SendLastPacket(true);
  FramesDecoder::SeekFrame(frame_id);
}

bool FramesDecoderGpu::ReadNextFrame(uint8_t *data, bool copy_to_output) {
  // No more frames in the file
  if (next_frame_idx_ == -1) {
    return false;
  }

  // Check if requested frame was buffered earlier
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == index_[next_frame_idx_].pts) {
      if (copy_to_output) {
        copyD2D(data, frame.frame_.data(), FrameSize());
      }
      LOG_LINE << "Read frame, index " << next_frame_idx_ << ", timestamp " <<
        std::setw(5) << frame.pts_ << ", current copy " << copy_to_output << std::endl;

      frame.pts_ = -1;

      ++next_frame_idx_;
      return true;
    }
  }

  current_copy_to_output_ = copy_to_output;
  current_frame_output_ = data;

  while (av_read_frame(av_state_->ctx_, av_state_->packet_) >= 0) {
    if (av_state_->packet_->stream_index != av_state_->stream_id_) {
      continue;
    }

    // Store pts from current packet to indicate,
    // that this frame is in the decoder
    piped_pts_.push(av_state_->packet_->pts);

    // Add header needed for NVDECODE to the packet
    if (filtered_packet_->data) {
      av_packet_unref(filtered_packet_);
    }
    DALI_ENFORCE(av_bsf_send_packet(bsfc_, av_state_->packet_) >= 0);
    DALI_ENFORCE(av_bsf_receive_packet(bsfc_, filtered_packet_) >= 0);

    // Prepare nv packet
    CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
    memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
    packet->payload = filtered_packet_->data;
    packet->payload_size = filtered_packet_->size;
    packet->flags = CUVID_PKT_TIMESTAMP;
    packet->timestamp = filtered_packet_->pts;

    // Send packet to the nv decoder
    frame_returned_ = false;
    CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

    if (frame_returned_) {
      ++next_frame_idx_;
      return true;
    }
  }

  DALI_ENFORCE(piped_pts_.size() == 1);

  SendLastPacket();
  next_frame_idx_ = -1;
  return true;
}

void FramesDecoderGpu::SendLastPacket(bool flush) {
  flush_ = flush;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));
  flush_ = false;

  if (flush) {
    // Clear frames buffer
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
      frame_buffer_[i].pts_ = -1;
    }

    // Clear piped pts
    while (piped_pts_.size() > 0) {
      piped_pts_.pop();
    }
  }
}

BufferedFrame& FramesDecoderGpu::FindEmptySlot() {
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == -1) {
      return frame;
    }
  }
  DALI_FAIL("Could not find empty slot in the frame buffer");
}

void FramesDecoderGpu::Reset() {
  SendLastPacket(true);
  FramesDecoder::Reset();
}

NvDecodeState::~NvDecodeState() {
  cuvidDestroyVideoParser(parser);
  cuvidDestroyDecoder(decoder);
}

FramesDecoderGpu::~FramesDecoderGpu() {
  av_packet_free(&filtered_packet_);
  av_bsf_free(&bsfc_);
}
}  // namespace dali
