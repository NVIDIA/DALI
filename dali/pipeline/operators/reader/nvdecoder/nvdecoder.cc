// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/reader/nvdecoder/nvdecoder.h"

#include <cuda_runtime.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <utility>

#include "dali/pipeline/operators/reader/nvdecoder/cuvideoparser.h"
#include "dali/util/cucontext.h"
#include "dali/util/dynlink_cuda.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/reader/nvdecoder/imgproc.h"
#include "dali/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h"

namespace dali {

NvDecoder::NvDecoder(int device_id,
                     const CodecParameters* codecpar,
                     AVRational time_base,
                     DALIImageType image_type,
                     DALIDataType dtype,
                     bool normalized)
    : device_id_(device_id), stream_(device_id, false, 0), codecpar_(codecpar),
      rgb_(image_type == DALI_RGB), dtype_(dtype), normalized_(normalized),
      device_(), context_(), parser_(), decoder_(),
      time_base_{time_base.num, time_base.den},
      frame_in_use_(32),  // 32 is cuvid's max number of decode surfaces
      recv_queue_(), frame_queue_(),
      current_recv_(), textures_(), done_(false) {
  if (!codecpar) {
    // TODO(spanev) explicit handling
    return;
  }

  DALI_ENFORCE(cuInitChecked(),
    "Failed to load libcuda.so. "
    "Check your library paths and if NVIDIA driver is installed correctly.");

  CUDA_CALL(cuDeviceGet(&device_, device_id_));

  char device_name[100];
  CUDA_CALL(cuDeviceGetName(device_name, 100, device_));
  LOG_LINE << "Using device: " << device_name << std::endl;

  context_ = CUContext(device_);
  if (!context_.initialized()) {
    DALI_FAIL("Problem initializing context, not initializing VideoDecoder");
  }

  auto codec = Codec::H264;
  switch (codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      codec = Codec::H264;
      break;

    case AV_CODEC_ID_HEVC:
      codec = Codec::HEVC;
      break;

    default:
      DALI_FAIL("Invalid codec for NvDecoder");
      return;
  }

  parser_.init(codec, this, 20, codecpar->extradata,
                      codecpar->extradata_size);
  if (!parser_.initialized()) {
    DALI_FAIL("Problem creating video parser");
    return;
  }
}

bool NvDecoder::initialized() const {
    return parser_.initialized();
}
NvDecoder::~NvDecoder() = default;

int NvDecoder::decode_av_packet(AVPacket* avpkt) {
  if (done_) return 0;

  CUVIDSOURCEDATAPACKET cupkt = {0};

  context_.push();

  if (avpkt && avpkt->size) {
      cupkt.payload_size = avpkt->size;
      cupkt.payload = avpkt->data;
      if (avpkt->pts != AV_NOPTS_VALUE) {
        cupkt.flags = CUVID_PKT_TIMESTAMP;
        if (time_base_.num && time_base_.den) {
          cupkt.timestamp = av_rescale_q(avpkt->pts, time_base_, nv_time_base_);
        } else {
          cupkt.timestamp = avpkt->pts;
        }
      }
  } else {
      cupkt.flags = CUVID_PKT_ENDOFSTREAM;
      // mark as flushing?
  }

  // parser_ will call handle_* callbacks after parsing
  CUDA_CALL(cuvidParseVideoData(parser_, &cupkt));
  return 0;
}

int NvDecoder::handle_sequence(void* user_data, CUVIDEOFORMAT* format) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_sequence_(format);
}

int NvDecoder::handle_decode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_decode_(pic_params);
}

int NvDecoder::handle_display(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_display_(disp_info);
}

int NvDecoder::handle_sequence_(CUVIDEOFORMAT* format) {
  frame_base_ = {static_cast<int>(format->frame_rate.denominator),
                  static_cast<int>(format->frame_rate.numerator)};

  // Prepare params and calls cuvidCreateDecoder
  return decoder_.initialize(format);
}

int NvDecoder::handle_decode_(CUVIDPICPARAMS* pic_params) {
  int total_wait = 0;
  constexpr auto sleep_period = 500;
  constexpr auto timeout_sec = 20;
  constexpr auto enable_timeout = false;
  while (frame_in_use_[pic_params->CurrPicIdx]) {
    if (enable_timeout &&
      total_wait++ > timeout_sec * 1000000 / sleep_period) {
      std::cout << device_id_ << ": Waiting for picture "
                << pic_params->CurrPicIdx
                << " to become available..." << std::endl;
      std::stringstream ss;
      ss << "Waited too long (" << timeout_sec << " seconds) "
          << "for decode output buffer to become available";
      DALI_FAIL(ss.str());
    }
    usleep(sleep_period);
    if (done_) return 0;
  }

  LOG_LINE << "Sending a picture for decode"
              << " size: " << pic_params->nBitstreamDataLen
              << " pic index: " << pic_params->CurrPicIdx
              << std::endl;

  // decoder_ operator () returns a CUvideodecoder
  CUDA_CALL(cuvidDecodePicture(decoder_, pic_params));
  return 1;
}

NvDecoder::MappedFrame::MappedFrame()
    : disp_info{nullptr}, valid_{false} {
}

NvDecoder::MappedFrame::MappedFrame(CUVIDPARSERDISPINFO* disp_info,
                                    CUvideodecoder decoder,
                                    CUstream stream)
    : disp_info{disp_info}, valid_{false}, decoder_(decoder), params_{0} {

  if (!disp_info->progressive_frame) {
    DALI_FAIL("Got an interlaced frame. We don't do interlaced frames.");
  }

  params_.progressive_frame = disp_info->progressive_frame;
  params_.top_field_first = disp_info->top_field_first;
  params_.second_field = 0;
  params_.output_stream = stream;

  CUDA_CALL(cuvidMapVideoFrame(decoder_, disp_info->picture_index,
                                &ptr_, &pitch_, &params_));
  valid_ = true;
}

NvDecoder::MappedFrame::MappedFrame(MappedFrame&& other)
    : disp_info(other.disp_info), valid_(other.valid_), decoder_(other.decoder_),
      ptr_(other.ptr_), pitch_(other.pitch_), params_(other.params_) {
  other.disp_info = nullptr;
  other.valid_ = false;
}

NvDecoder::MappedFrame::~MappedFrame() {
  if (valid_) {
    CUDA_CALL(cuvidUnmapVideoFrame(decoder_, ptr_));
  }
}

uint8_t* NvDecoder::MappedFrame::get_ptr() const {
  return reinterpret_cast<uint8_t*>(ptr_);
}

unsigned int NvDecoder::MappedFrame::get_pitch() const {
  return pitch_;
}

NvDecoder::TextureObject::TextureObject() : valid_{false} {
}

NvDecoder::TextureObject::TextureObject(const cudaResourceDesc* pResDesc,
                                        const cudaTextureDesc* pTexDesc,
                                        const cudaResourceViewDesc* pResViewDesc)
    : valid_{false}
{
  CUDA_CALL(cudaCreateTextureObject(&object_, pResDesc, pTexDesc, pResViewDesc));
  valid_ = true;
}

NvDecoder::TextureObject::~TextureObject() {
    if (valid_) {
        cudaDestroyTextureObject(object_);
    }
}

NvDecoder::TextureObject::TextureObject(NvDecoder::TextureObject&& other)
    : valid_{other.valid_}, object_{other.object_}
{
  other.valid_ = false;
}

NvDecoder::TextureObject& NvDecoder::TextureObject::operator=(NvDecoder::TextureObject&& other) {
  valid_ = other.valid_;
  object_ = other.object_;
  other.valid_ = false;
  return *this;
}

NvDecoder::TextureObject::operator cudaTextureObject_t() const {
  if (valid_) {
      return object_;
  } else {
      return cudaTextureObject_t{};
  }
}

// Callback called by the driver decoder one a frame has been decoded
int NvDecoder::handle_display_(CUVIDPARSERDISPINFO* disp_info) {
  auto frame = av_rescale_q(disp_info->timestamp,
                            nv_time_base_, frame_base_);

  if (current_recv_.count <= 0) {
    if (recv_queue_.empty()) {
      LOG_LINE << "Ditching frame " << frame << " since "
              << "the receive queue is empty." << std::endl;
      return 1;
    }
    LOG_LINE << "Moving on to next request, " << recv_queue_.size()
                << " reqs left" << std::endl;
    current_recv_ = recv_queue_.pop();
  }

  if (done_) return 0;

  if (current_recv_.count <= 0) {
    // a new req with count <= 0 probably means we are finishing
    // up and should just ditch this frame
    LOG_LINE << "Ditching frame " << frame << "since current_recv_.count <= 0" << std::endl;
    return 1;
  }

  if (frame != current_recv_.frame) {
    // TODO(spanev) This definitely needs better error handling...
    // Add exception? Directly or after countdown treshold?
    LOG_LINE << "Ditching frame " << frame << " since we are waiting for "
                << "frame " << current_recv_.frame << std::endl;
    return 1;
  }

  LOG_LINE << "\e[1mGoing ahead with frame " << frame
              << " wanted count: " << current_recv_.count
              << " disp_info->picture_index: " << disp_info->picture_index
              << "\e[0m" << std::endl;

  current_recv_.frame++;
  current_recv_.count--;

  frame_in_use_[disp_info->picture_index] = true;
  frame_queue_.push(disp_info);
  return 1;
}

int NvDecoder::decode_packet(AVPacket* pkt) {
  switch (codecpar_->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
    case AVMEDIA_TYPE_VIDEO:
      return decode_av_packet(pkt);

    default:
      DALI_FAIL("Got to decode_packet in a decoder that is not "
                "for an audio, video, or subtitle stream.");
  }
  return -1;
}


void NvDecoder::push_req(FrameReq req) {
  recv_queue_.push(std::move(req));
}

void NvDecoder::receive_frames(SequenceWrapper& sequence) {
  LOG_LINE << "Sequence pushed with " << sequence.count << " frames" << std::endl;

  context_.push();
  for (int i = 0; i < sequence.count; ++i) {
      LOG_LINE << "popping frame (" << i << "/" << sequence.count << ") "
               << frame_queue_.size() << " reqs left"
               << std::endl;

      auto* frame_disp_info = frame_queue_.pop();
      if (done_) break;
      auto frame = MappedFrame{frame_disp_info, decoder_, stream_};
      if (done_) break;
      convert_frame(frame, sequence, i);
    }
  record_sequence_event_(sequence);
}

// We assume here that a pointer and scale_method
// uniquely identifies a texture
const NvDecoder::TextureObjects&
NvDecoder::get_textures(uint8_t* input, unsigned int input_pitch,
                        uint16_t input_width, uint16_t input_height,
                        ScaleMethod scale_method) {
  auto tex_id = std::make_tuple(input, scale_method);
  auto tex = textures_.find(tex_id);
  if (tex != textures_.end()) {
    return tex->second;
  }
  TextureObjects objects;
  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0]   = cudaAddressModeClamp;
  tex_desc.addressMode[1]   = cudaAddressModeClamp;
  if (scale_method == ScaleMethod_Nearest) {
      tex_desc.filterMode   = cudaFilterModePoint;
  } else {
      tex_desc.filterMode   = cudaFilterModeLinear;
  }
  tex_desc.readMode         = cudaReadModeNormalizedFloat;
  tex_desc.normalizedCoords = 0;

  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypePitch2D;
  res_desc.res.pitch2D.devPtr = input;
  res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
  res_desc.res.pitch2D.width = input_width;
  res_desc.res.pitch2D.height = input_height;
  res_desc.res.pitch2D.pitchInBytes = input_pitch;

  objects.luma = TextureObject{&res_desc, &tex_desc, nullptr};

  tex_desc.addressMode[0]   = cudaAddressModeClamp;
  tex_desc.addressMode[1]   = cudaAddressModeClamp;
  tex_desc.filterMode       = cudaFilterModeLinear;
  tex_desc.readMode         = cudaReadModeNormalizedFloat;
  tex_desc.normalizedCoords = 0;

  res_desc.resType = cudaResourceTypePitch2D;
  res_desc.res.pitch2D.devPtr = input + (input_height * input_pitch);
  res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
  res_desc.res.pitch2D.width = input_width;
  res_desc.res.pitch2D.height = input_height / 2;
  res_desc.res.pitch2D.pitchInBytes = input_pitch;

  objects.chroma = TextureObject{&res_desc, &tex_desc, nullptr};

  auto p = textures_.emplace(tex_id, std::move(objects));
  if (!p.second) {
    DALI_FAIL("Unable to cache a new texture object.");
  }
  return p.first->second;
}

void NvDecoder::convert_frame(const MappedFrame& frame, SequenceWrapper& sequence,
                              int index) {
  auto input_width = decoder_.width();
  auto input_height = decoder_.height();

  auto output_idx = index;
  // TODO(spanev) Add ScaleMethod choice
  auto& textures = this->get_textures(frame.get_ptr(),
                                      frame.get_pitch(),
                                      input_width,
                                      input_height,
                                      ScaleMethod_Nearest);
  if (dtype_ == DALI_UINT8) {
    process_frame<uint8>(textures.chroma, textures.luma,
                  sequence,
                  output_idx, stream_,
                  input_width, input_height,
                  rgb_, normalized_);
  } else {  // dtype_ == DALI_FLOAT
    process_frame<float>(textures.chroma, textures.luma,
                  sequence,
                  output_idx, stream_,
                  input_width, input_height,
                  rgb_, normalized_);
  }

  frame_in_use_[frame.disp_info->picture_index] = false;
}

void NvDecoder::finish() {
  done_ = true;
  recv_queue_.shutdown();
  frame_queue_.shutdown();
}

void NvDecoder::record_sequence_event_(SequenceWrapper& sequence) {
  sequence.set_started(stream_);
}

}  // namespace dali
