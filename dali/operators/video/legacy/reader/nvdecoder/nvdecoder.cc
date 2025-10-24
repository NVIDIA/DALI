// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/legacy/reader/nvdecoder/nvdecoder.h"
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/operators/video/legacy/reader/nvdecoder/cuvideoparser.h"
#include "dali/operators/video/color_space.h"

namespace dali {

static constexpr int kNvcuvid_success = 1;
static constexpr int kNvcuvid_failure = 0;

NvDecoder::NvDecoder(int device_id,
                     const CodecParameters* codecpar,
                     DALIImageType image_type,
                     DALIDataType dtype,
                     bool normalized,
                     int max_height,
                     int max_width,
                     int additional_decode_surfaces)
    : device_id_(device_id),
      rgb_(image_type == DALI_RGB), dtype_(dtype), normalized_(normalized),
      device_(), parser_(), decoder_(max_height, max_width, additional_decode_surfaces),
      frame_in_use_(32),  // 32 is cuvid's max number of decode surfaces
      frame_full_range_(32),  // 32 is cuvid's max number of decode surfaces
      recv_queue_(), frame_queue_(),
      current_recv_(), req_ready_(VidReqStatus::REQ_READY), stop_(false) {

  DALI_ENFORCE(cuInitChecked(),
    "Failed to load libcuda.so. "
    "Check your library paths and if NVIDIA driver is installed correctly.");

  // This is a workaround for an issue with nvcuvid in drivers >460 and < 470.21 where concurrent
  // use on default context and non-default streams may lead to memory corruption.
  bool use_default_stream = false;
#if NVML_ENABLED
  {
    nvml_handle_ = nvml::NvmlInstance::CreateNvmlInstance();
    static float driver_version = nvml::GetDriverVersion();
    if (driver_version > 460 && driver_version < 470.21)
      use_default_stream = true;
  }
#else
  {
    int driver_cuda_version = 0;
    CUDA_CALL(cuDriverGetVersion(&driver_cuda_version));
    if (driver_cuda_version >= 11030 && driver_cuda_version < 11040)
      use_default_stream = true;
  }
#endif
  if (use_default_stream) {
    DALI_WARN_ONCE("Warning: Decoding on a default stream. Performance may be affected.");
    stream_ = {};
  } else {
    stream_ = CUDAStreamPool::instance().Get(device_id);
  }

  CUDA_CALL(cuDeviceGet(&device_, device_id_));

  char device_name[100];
  CUDA_CALL(cuDeviceGetName(device_name, 100, device_));
  LOG_LINE << "Using device: " << device_name << std::endl;
  DeviceGuard g(device_id_);

  auto codec = Codec::H264;
  switch (codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      codec = Codec::H264;
      break;

    case AV_CODEC_ID_HEVC:
      codec = Codec::HEVC;
      break;

    case AV_CODEC_ID_MPEG4:
      codec = Codec::MPEG4;
      break;

    case AV_CODEC_ID_VP9:
      codec = Codec::VP9;
      break;

    case AV_CODEC_ID_VP8:
      codec = Codec::VP8;
      break;


    case AV_CODEC_ID_MJPEG:
      codec = Codec::MJPEG;
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

NvDecoder::~NvDecoder() {}

VidReqStatus NvDecoder::decode_av_packet(AVPacket* avpkt, int64_t start_time,
                                         AVRational stream_base) {
  if (stop_) {
    LOG_LINE << "NvDecoder::stop_ requested" << std::endl;
    return VidReqStatus::REQ_READY;
  }

  CUVIDSOURCEDATAPACKET cupkt = {0};

  DeviceGuard g(device_id_);

  if (avpkt && avpkt->size) {
      cupkt.payload_size = avpkt->size;
      cupkt.payload = avpkt->data;
      if (avpkt->pts != AV_NOPTS_VALUE) {
        cupkt.flags = CUVID_PKT_TIMESTAMP;
        if (stream_base.num && stream_base.den) {
          cupkt.timestamp = av_rescale_q(avpkt->pts - start_time, stream_base, nv_time_base_);
        } else {
          cupkt.timestamp = avpkt->pts - start_time;
        }
      }
  } else {
      cupkt.flags = CUVID_PKT_ENDOFSTREAM;
      // mark as flushing?
  }

  // parser_ will call handle_* callbacks after parsing
  auto ret = cuvidParseVideoData(parser_, &cupkt);
  if (!captured_exception_) {
    // throw only if we haven't captured any other exception before which is probably processed
    // right now and we don't want to throw exception in exception
    NVCUVID_CALL(ret);
  }
  return req_ready_;
}

int NvDecoder::handle_sequence(void* user_data, CUVIDEOFORMAT* format) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_sequence_(format);
}

int NvDecoder::handle_decode(void* user_data, CUVIDPICPARAMS* pic_params) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_decode_(pic_params);
}

int NvDecoder::handle_display(void* user_data, CUVIDPARSERDISPINFO* disp_info) {
  auto decoder = reinterpret_cast<NvDecoder*>(user_data);
  return decoder->handle_display_(disp_info);
}

// Prepare params and calls cuvidCreateDecoder
int NvDecoder::handle_sequence_(CUVIDEOFORMAT* format) {
  int ret = kNvcuvid_failure;
  try {
    // utilize the NVDEC parser to determine if the video is full range
    recv_queue_.front().full_range =
        static_cast<bool>(format->video_signal_description.video_full_range_flag);
    ret = decoder_.initialize(format);
  } catch (...) {
    ERROR_LOG << "Unable to decode file " << recv_queue_.peek().filename << '\n';
    stop_ = true;
    captured_exception_ = std::current_exception();
    // Main thread is waiting on frame_queue_
    frame_queue_.shutdown();
  }
  return ret;
}

int NvDecoder::handle_decode_(CUVIDPICPARAMS* pic_params) {
  int total_wait = 0;
  constexpr auto sleep_period = 500;
  constexpr auto timeout_sec = 20;
  constexpr auto enable_timeout = false;

  // If something went wrong during init we exit directly
  if (stop_) return kNvcuvid_failure;

  while (frame_in_use_[pic_params->CurrPicIdx]) {
    if (enable_timeout &&
      total_wait++ > timeout_sec * 1000000 / sleep_period) {
      LOG_LINE << device_id_ << ": Waiting for picture "
               << pic_params->CurrPicIdx
               << " to become available..." << std::endl;
      std::stringstream ss;
      ss << "Waited too long (" << timeout_sec << " seconds) "
         << "for decode output buffer to become available";
      DALI_FAIL(ss.str());
    }
    usleep(sleep_period);
    if (stop_) return kNvcuvid_failure;
  }

  LOG_LINE << "Sending a picture for decode"
           << " size: " << pic_params->nBitstreamDataLen
           << " pic index: " << pic_params->CurrPicIdx
           << std::endl;

  // decoder_ operator () returns a CUvideodecoder
  NVCUVID_CALL(cuvidDecodePicture(decoder_, pic_params));
  return kNvcuvid_success;
}

NvDecoder::MappedFrame::MappedFrame(CUVIDPARSERDISPINFO* disp_info,
                                    CUvideodecoder decoder,
                                    CUstream stream)
    : disp_info{disp_info}, valid_{false}, decoder_(decoder), params_{0} {

  assert(disp_info);

  if (!disp_info->progressive_frame) {
    DALI_FAIL("Got an interlaced frame. We don't do interlaced frames.");
  }

  params_.progressive_frame = disp_info->progressive_frame;
  params_.top_field_first = disp_info->top_field_first;
  params_.second_field = 0;
  params_.output_stream = stream;

  NVCUVID_CALL(cuvidMapVideoFrame(decoder_, disp_info->picture_index,
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
    NVCUVID_CALL(cuvidUnmapVideoFrame(decoder_, ptr_));
  }
}

uint8_t* NvDecoder::MappedFrame::get_ptr() const {
  return reinterpret_cast<uint8_t*>(ptr_);
}

unsigned int NvDecoder::MappedFrame::get_pitch() const {
  return pitch_;
}

// Callback called by the driver decoder once a frame has been decoded
int NvDecoder::handle_display_(CUVIDPARSERDISPINFO* disp_info) {
  auto frame = av_rescale_q(disp_info->timestamp,
                            nv_time_base_, current_recv_.frame_base);

  // it means that any frame has been decoded
  if (req_ready_ == VidReqStatus::REQ_NOT_STARTED) {
    req_ready_ = VidReqStatus::REQ_IN_PROGRESS;
  }

  if (current_recv_.count <= 0) {
    if (recv_queue_.empty()) {
      LOG_LINE << "Ditching frame " << frame << " since "
               << "the receive queue is empty." << std::endl;
      return kNvcuvid_success;
    }
    LOG_LINE << "Moving on to next request, " << recv_queue_.size()
                << " reqs left" << std::endl;
    current_recv_ = recv_queue_.pop();
    frame =  av_rescale_q(disp_info->timestamp,
                          nv_time_base_, current_recv_.frame_base);
  }

  if (stop_) return kNvcuvid_failure;

  if (current_recv_.count <= 0) {
    // a new req with count <= 0 probably means we are finishing
    // up and should just ditch this frame
    LOG_LINE << "Ditching frame " << frame << "since current_recv_.count <= 0" << std::endl;
    return kNvcuvid_success;
  }

  if (frame < current_recv_.frame) {
    // TODO(spanev) This definitely needs better error handling...
    // Add exception? Directly or after countdown treshold?
    LOG_LINE << "Ditching frame " << frame << " since we are waiting for "
                << "frame " << current_recv_.frame << std::endl;
    return kNvcuvid_success;
  } else if  (frame > current_recv_.frame) {
    LOG_LINE << "Receive frame " << frame << " that is pas the exptected "
                << "frame " << current_recv_.frame << std::endl;
    req_ready_ = VidReqStatus::REQ_ERROR;
    stop_ = true;
    // Main thread is waiting on frame_queue_
    frame_queue_.shutdown();
    return kNvcuvid_failure;
  }

  LOG_LINE << "\e[1mGoing ahead with frame " << frame
              << " wanted count: " << current_recv_.count
              << " disp_info->picture_index: " << disp_info->picture_index
              << "\e[0m" << std::endl;

  current_recv_.frame += current_recv_.stride;
  current_recv_.count -= current_recv_.stride;

  frame_in_use_[disp_info->picture_index] = true;
  frame_full_range_[disp_info->picture_index] = current_recv_.full_range;
  frame_queue_.push(disp_info);
  if (current_recv_.count <= 0) {
    req_ready_ = VidReqStatus::REQ_READY;
  }
  return kNvcuvid_success;
}

VidReqStatus NvDecoder::decode_packet(AVPacket* pkt, int64_t start_time, AVRational stream_base,
                             const CodecParameters* codecpar) {
  AVMediaType codec_type = AVMEDIA_TYPE_VIDEO;
  // if they are null we are flushing the decoder and we don't want the bellow check
  if (pkt && codecpar) {
    codec_type = codecpar->codec_type;
  }
  switch (codec_type) {
    case AVMEDIA_TYPE_AUDIO:
    case AVMEDIA_TYPE_VIDEO:
      return decode_av_packet(pkt, start_time, stream_base);

    default:
      DALI_FAIL("Got to decode_packet in a decoder that is not "
                "for an audio, video, or subtitle stream.");
  }
  return VidReqStatus::REQ_READY;
}

void NvDecoder::push_req(FrameReq req) {
  req_ready_ = VidReqStatus::REQ_NOT_STARTED;
  recv_queue_.push(std::move(req));
}

void NvDecoder::receive_frames(SequenceWrapper& sequence) {
  LOG_LINE << "Sequence pushed with " << sequence.count << " frames" << std::endl;

  DeviceGuard g(device_id_);
  for (int i = 0; i < sequence.count; ++i) {
      LOG_LINE << "popping frame (" << i << "/" << sequence.count << ") "
               << frame_queue_.size() << " reqs left" << std::endl;

      auto* frame_disp_info = frame_queue_.pop();
      if (stop_) break;
      auto frame = MappedFrame{frame_disp_info, decoder_, stream_};
      sequence.timestamps.push_back(frame_disp_info->timestamp * av_q2d(
            nv_time_base_));
      if (stop_) break;
      auto is_full_range = frame_full_range_[frame.disp_info->picture_index];
      auto conversion_type =  rgb_ ?
                        is_full_range ? VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE :
                                        VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB :
                        VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE;
      auto frame_stride =
              static_cast<ptrdiff_t>(i) * sequence.height * sequence.width * sequence.channels;
      TYPE_SWITCH(dtype_, type2id, OutType, NVDECODER_SUPPORTED_TYPES, (
        auto* tensor_out = sequence.sequence.mutable_data<OutType>() + frame_stride;
        VideoColorSpaceConversion(
          reinterpret_cast<OutType *>(tensor_out), sequence.width * sequence.channels,
          reinterpret_cast<const uint8_t *>(frame.get_ptr()), static_cast<int>(frame.get_pitch()),
          decoder_.height(),
          decoder_.width(),
          conversion_type,
          normalized_,
          stream_);
      ), DALI_FAIL(make_string("Unsupported type: ", dtype_)));
      // synchronize before MappedFrame is destroyed and cuvidUnmapVideoFrame is called
      CUDA_CALL(cudaStreamSynchronize(stream_));
      frame_in_use_[frame.disp_info->picture_index] = false;
  }
  if (captured_exception_)
    std::rethrow_exception(captured_exception_);
  if (sequence.count < sequence.max_count) {
    auto data_size = sequence.count * volume(sequence.frame_shape());
    auto pad_size = (sequence.max_count - sequence.count) * volume(sequence.frame_shape()) *
                     dali::TypeTable::GetTypeInfo(sequence.dtype).size();
    TYPE_SWITCH(dtype_, type2id, OutputType, NVDECODER_SUPPORTED_TYPES, (
      cudaMemsetAsync(sequence.sequence.mutable_data<OutputType>() + data_size, 0, pad_size,
                      stream_);
    ), DALI_FAIL(make_string("Not supported output type:", dtype_, // NOLINT
        "Only DALI_UINT8 and DALI_FLOAT are supported as the decoder outputs.")););
  }
  record_sequence_event_(sequence);
}

void NvDecoder::finish() {
  stop_ = true;
  recv_queue_.shutdown();
  frame_queue_.shutdown();
}

void NvDecoder::record_sequence_event_(SequenceWrapper& sequence) {
  sequence.set_started(stream_);
}

}  // namespace dali
