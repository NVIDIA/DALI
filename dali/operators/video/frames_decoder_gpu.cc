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

#include "dali/operators/video/frames_decoder_gpu.h"
#include <cuda.h>
#include <unistd.h>
#include <string>
#include <memory>
#include <iomanip>
#include <map>
#include <mutex>
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/video/color_space.h"
#include "dali/core/static_switch.h"
#include "dali/operators/video/video_utils.h"

namespace dali {

namespace frame_dec_gpu_impl {

const char *chroma_to_string(cudaVideoChromaFormat in) {
  switch (in) {
    case cudaVideoChromaFormat_Monochrome:
      return "Monochrome";
    case cudaVideoChromaFormat_420:
      return "YUV 4:2:0";
    case cudaVideoChromaFormat_422:
      return "YUV 4:2:2";
    case cudaVideoChromaFormat_444:
      return "YUV 4:4:4";
    default:
      return "Unknown chroma format";
  }
}

const char *codec_to_string(cudaVideoCodec in) {
  switch (in) {
    case cudaVideoCodec_MPEG1:
      return "MPEG-1";
    case cudaVideoCodec_MPEG2:
      return "MPEG-2";
    case cudaVideoCodec_MPEG4:
      return "MPEG-4";
    case cudaVideoCodec_VC1:
      return "VC1";
    case cudaVideoCodec_H264:
      return "H.264";
    case cudaVideoCodec_JPEG:
      return "JPEG";
    case cudaVideoCodec_H264_SVC:
      return "H.264-SVC";
    case cudaVideoCodec_H264_MVC:
      return "H.264-MVC";
    case cudaVideoCodec_HEVC:
      return "HEVC";
    case cudaVideoCodec_VP8:
      return "VP8";
    case cudaVideoCodec_VP9:
      return "VP9";
    case cudaVideoCodec_AV1:
      return "AV1";
    default:
      return "Unknown codec type";
  }
}

class NVDECCache {
 public:
    static NVDECCache &GetCache(int device_id = -1) {
      static NVDECCache cache_inst[32];
      if (device_id == -1) {
        CUDA_CALL(cudaGetDevice(&device_id));
      }
      return cache_inst[device_id];
    }

    static NVDECLease GetDecoderFromCache(CUVIDEOFORMAT *video_format, int device_id = -1) {
      if (device_id == -1) {
        CUDA_CALL(cudaGetDevice(&device_id));
      }
      return GetCache(device_id).GetDecoder(video_format, device_id);
    }

    void ReturnDecoder(DecInstance *decoder) {
      std::unique_lock lock(access_lock);
      auto range = dec_cache.equal_range(decoder->codec_type);
      for (auto it = range.first; it != range.second; ++it) {
        if (&it->second == decoder) {
          it->second.used = false;
          return;
        }
      }
      DALI_FAIL("Cannot return decoder that is not from the cache");
    }

 private:
    NVDECCache() {}

    ~NVDECCache() {
      std::lock_guard lock(access_lock);
      for (auto &it : dec_cache) {
        cuvidDestroyDecoder(it.second.decoder);
      }
    }

    NVDECLease GetDecoder(CUVIDEOFORMAT *video_format, int device_id) {
      std::unique_lock lock(access_lock);

      auto codec_type = video_format->codec;
      unsigned height =  video_format->coded_height;
      unsigned width = video_format->coded_width;
      auto num_decode_surfaces = video_format->min_num_decode_surfaces;
      auto chroma_format = video_format->chroma_format;
      auto bit_depth_luma_minus8 = video_format->bit_depth_luma_minus8;

      if (num_decode_surfaces == 0)
        num_decode_surfaces = 20;
      auto range = dec_cache.equal_range(codec_type);
      codec_map::iterator best_match = range.second;
      for (auto it = range.first; it != range.second; ++it) {
        if (best_match == range.second && it->second.used == false) {
          best_match = it;
        }
        if (it->second.used == false && it->second.height == height &&
            it->second.width == width && it->second.num_decode_surfaces == num_decode_surfaces &&
            it->second.chroma_format == chroma_format &&
            it->second.bit_depth_luma_minus8 == bit_depth_luma_minus8) {
          it->second.used = true;
          assert(it->second.device_id == device_id);
          return NVDECLease(this, &it->second);
        }
      }
      // reconfigure needs ulTargetHeight and ulTargetWidth set to the upper bound of the video
      // resolution. Hard to know them ahead of time and setting too much slow things down
#ifdef ENABLE_NVDEC_RECONFIGURE
      if (best_match != range.second && cuvidIsSymbolAvailable("cuvidReconfigureDecoder") &&
          best_match->second.max_width <= width && best_match->second.max_height <= height &&
          best_match->second.chroma_format == chroma_format &&
          best_match->second.bit_depth_luma_minus8 == bit_depth_luma_minus8) {
        best_match->second.used = true;
        lock.unlock();
        CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };
        reconfigParams.ulTargetWidth = reconfigParams.ulWidth = width;
        reconfigParams.ulTargetHeight = reconfigParams.ulHeight = height;
        reconfigParams.ulNumDecodeSurfaces = num_decode_surfaces;
        best_match->second.height = height;
        best_match->second.width = width;
        best_match->second.num_decode_surfaces = num_decode_surfaces;
        CUDA_CALL(cuvidReconfigureDecoder(best_match->second.decoder, &reconfigParams));
        return NVDECLease(this, &best_match->second);
      }
#endif
      lock.unlock();
      auto caps = CUVIDDECODECAPS{};
      caps.eCodecType = codec_type;
      caps.eChromaFormat = chroma_format;
      caps.nBitDepthMinus8 = bit_depth_luma_minus8;
      CUDA_CALL(cuvidGetDecoderCaps(&caps));
      DALI_ENFORCE(caps.bIsSupported,
                   make_string("Codec configuration not supported on this GPU. ",
                   "Codec: ", codec_to_string(codec_type),
                   ", chroma format: ", chroma_to_string(chroma_format),
                   ", bit depth: ", bit_depth_luma_minus8 + 8));

      DALI_ENFORCE(width >= caps.nMinWidth  && height >= caps.nMinHeight,
                   make_string("Video is too small in at least one dimension. Provided: ",
                   width , "x", height, " vs supported:", caps.nMinWidth, "x", caps.nMinHeight));

      DALI_ENFORCE(width <= caps.nMaxWidth && height <= caps.nMaxHeight,
                   make_string("Video is too large in at least one dimension. Provided: ",
                   width , "x", height, " vs supported:", caps.nMaxWidth, "x", caps.nMaxHeight));

      DALI_ENFORCE(width * height / 256 <= caps.nMaxMBCount,
                   make_string("Video is too large (too many macroblocks). ",
                   "Provided (width * height / 256): ",
                   width * height / 256, " vs supported:", caps.nMaxMBCount));

      CUVIDDECODECREATEINFO decoder_info;
      memset(&decoder_info, 0, sizeof(CUVIDDECODECREATEINFO));

      decoder_info.bitDepthMinus8 = bit_depth_luma_minus8;
      decoder_info.ChromaFormat = chroma_format;
      decoder_info.CodecType = codec_type;
      decoder_info.ulHeight = height;
      decoder_info.ulWidth = width;
      // creating the decoder with ulTargetHeight and ulTargetWidth that are above what is really
      // needed slow things down
#ifdef ENABLE_NVDEC_RECONFIGURE
      // assume max width and calculate ulMaxHeight
      unsigned max_height = caps.nMaxHeight;
      unsigned max_width = caps.nMaxMBCount / caps.nMaxHeight * 256;
      if (max_height < height || max_width < width) {
        max_height = height;
        max_width = caps.nMaxMBCount / height * 256;
      }
#else
      unsigned max_height = height;
      unsigned max_width = width;
#endif
      decoder_info.ulMaxHeight = max_height;
      decoder_info.ulMaxWidth = max_width;
      decoder_info.ulTargetHeight = video_format->display_area.bottom -
                                    video_format->display_area.top;
      decoder_info.ulTargetWidth = video_format->display_area.right -
                                   video_format->display_area.left;
      decoder_info.ulNumDecodeSurfaces = num_decode_surfaces;
      decoder_info.ulNumOutputSurfaces = 2;

      auto& area = decoder_info.display_area;
      area.left   = video_format->display_area.left;
      area.right  = video_format->display_area.right;
      area.top    = video_format->display_area.top;
      area.bottom = video_format->display_area.bottom;
      DecInstance decoder_inst = {};

      CUDA_CALL(cuvidCreateDecoder(&(decoder_inst.decoder), &decoder_info));
      decoder_inst.height = decoder_info.ulTargetHeight;
      decoder_inst.width = decoder_info.ulTargetWidth;
      decoder_inst.max_height = decoder_info.ulMaxHeight;
      decoder_inst.max_width = decoder_info.ulMaxWidth;
      decoder_inst.num_decode_surfaces = num_decode_surfaces;
      decoder_inst.codec_type = codec_type;
      decoder_inst.chroma_format = chroma_format;
      decoder_inst.bit_depth_luma_minus8 = bit_depth_luma_minus8;
      decoder_inst.used = true;
      decoder_inst.device_id = device_id;

      lock.lock();
      auto inserted = dec_cache.insert({codec_type, decoder_inst});
      if (dec_cache.size() > CACHE_SIZE_LIMIT) {
        for (auto it = dec_cache.begin(); it != dec_cache.end(); ++it) {
          if (it->second.used == false) {
            auto decoder = it->second.decoder;
            dec_cache.erase(it);
            lock.unlock();
            cuvidDestroyDecoder(decoder);
            break;
          }
        }
      }
      return NVDECLease(this, &inserted->second);
    }

    using codec_map = std::multimap<cudaVideoCodec, DecInstance>;
    codec_map dec_cache;
    std::mutex access_lock;

    static constexpr int CACHE_SIZE_LIMIT = 100;
};

void NVDECLease::Return() {
  if (decoder) {
    owner->ReturnDecoder(decoder);
    owner = nullptr;
    decoder = nullptr;
  }
}

int process_video_sequence(void *user_data, CUVIDEOFORMAT *video_format) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);
  frames_decoder->InitGpuDecoder(video_format);

  return video_format->min_num_decode_surfaces;
}

int process_picture_decode(void *user_data, CUVIDPICPARAMS *picture_params) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);

  return frames_decoder->ProcessPictureDecode(picture_params);
}

int handle_picture_display(void *user_data, CUVIDPARSERDISPINFO *picture_display_info) {
  FramesDecoderGpu *frames_decoder = static_cast<FramesDecoderGpu*>(user_data);

  return frames_decoder->HandlePictureDisplay(picture_display_info);
}

}  // namespace frame_dec_gpu_impl

void FramesDecoderGpu::InitBitStreamFilter() {
  const AVBitStreamFilter *bsf = nullptr;
  const char* filtername = nullptr;
  switch (codec_params_->codec_id) {
  case AVCodecID::AV_CODEC_ID_H264:
    if  (!strcmp(ctx_->iformat->long_name, "QuickTime / MOV") ||
         !strcmp(ctx_->iformat->long_name, "FLV (Flash Video)") ||
         !strcmp(ctx_->iformat->long_name, "Matroska / WebM") ||
         !strcmp(ctx_->iformat->long_name, "raw H.264 video")) {
      filtername = "h264_mp4toannexb";
    }
    break;
  case AVCodecID::AV_CODEC_ID_HEVC:
    if  (!strcmp(ctx_->iformat->long_name, "QuickTime / MOV") ||
         !strcmp(ctx_->iformat->long_name, "FLV (Flash Video)") ||
         !strcmp(ctx_->iformat->long_name, "Matroska / WebM") ||
         !strcmp(ctx_->iformat->long_name, "raw HEVC video")) {
      filtername = "hevc_mp4toannexb";
    }
    break;
  case AVCodecID::AV_CODEC_ID_MPEG4:
    if  (!strcmp(ctx_->iformat->name, "avi")) {
      filtername = "mpeg4_unpack_bframes";
    }
    break;
  case AVCodecID::AV_CODEC_ID_MJPEG:
  case AVCodecID::AV_CODEC_ID_VP8:
  case AVCodecID::AV_CODEC_ID_VP9:
  case AVCodecID::AV_CODEC_ID_AV1:
    break;  // No bit stream filter needed for MJPEG, VP8, VP9, AV1
  default:
    DALI_FAIL(make_string(
      "Could not find suitable bit stream filter for codec: ",
      avcodec_get_name(codec_params_->codec_id)));
  }

  if (filtername != nullptr) {
    auto bsf = av_bsf_get_by_name(filtername);
    DALI_ENFORCE(bsf, "Error finding bit stream filter.");
    DALI_ENFORCE(av_bsf_alloc(bsf, &bsfc_) >= 0,
                 "Unable to allocate bit stream filter");
  } else {
    DALI_ENFORCE(av_bsf_get_null_filter(&bsfc_) >= 0,
                 "Error creating pass-through filter.");
  }

  DALI_ENFORCE(
    avcodec_parameters_copy(bsfc_->par_in,
                            ctx_->streams[stream_id_]->codecpar) >= 0,
    "Unable to copy bit stream filter parameters");
  DALI_ENFORCE(
    av_bsf_init(bsfc_) >= 0,
    "Unable to initialize bit stream filter");
}

cudaVideoCodec FramesDecoderGpu::GetCodecType(AVCodecID codec_id) const {
  switch (codec_id) {
    case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
    case AV_CODEC_ID_MPEG4: return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_VP8: return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9: return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG: return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1: return cudaVideoCodec_AV1;
    default: {
      DALI_FAIL(make_string("Unsupported codec type ", avcodec_get_name(codec_id)));
    }
  }
}

void FramesDecoderGpu::InitGpuDecoder(CUVIDEOFORMAT *video_format) {
  if (!nvdecode_state_->decoder) {
    bool is_full_range = video_format->video_signal_description.video_full_range_flag;
    conversion_type_ = image_type_ == DALI_RGB ?
                           is_full_range ? VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE :
                                           VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB :
                           VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE;
    nvdecode_state_->decoder = frame_dec_gpu_impl::NVDECCache::GetDecoderFromCache(video_format);
  }
}

void FramesDecoderGpu::InitGpuParser() {
  nvdecode_state_ = std::make_unique<NvDecodeState>();

  InitBitStreamFilter();

  filtered_packet_.reset(av_packet_alloc());
  DALI_ENFORCE(filtered_packet_,
               make_string("Could not allocate av packet for \"", Filename(), "\""));

  auto codec_type = GetCodecType(codec_params_->codec_id);

  // Create nv parser
  CUVIDPARSERPARAMS parser_info;
  CUVIDEOFORMATEX parser_extinfo;
  memset(&parser_info, 0, sizeof(CUVIDPARSERPARAMS));
  parser_info.CodecType = codec_type;
  parser_info.ulMaxNumDecodeSurfaces = num_decode_surfaces_;
  parser_info.ulMaxDisplayDelay = 0;
  parser_info.pUserData = this;
  parser_info.pfnSequenceCallback = frame_dec_gpu_impl::process_video_sequence;
  parser_info.pfnDecodePicture = frame_dec_gpu_impl::process_picture_decode;
  parser_info.pfnDisplayPicture = nullptr;

  auto extradata = ctx_->streams[stream_id_]->codecpar->extradata;
  auto extradata_size = ctx_->streams[stream_id_]->codecpar->extradata_size;

  memset(&parser_extinfo, 0, sizeof(parser_extinfo));
  parser_info.pExtVideoInfo = &parser_extinfo;
  if (extradata_size > 0) {
    auto hdr_size = std::min(sizeof(parser_extinfo.raw_seqhdr_data),
                             static_cast<std::size_t>(extradata_size));
    parser_extinfo.format.seqhdr_data_length = hdr_size;
    memcpy(parser_extinfo.raw_seqhdr_data, extradata, hdr_size);
  }

  nvdecode_state_->parser = frame_dec_gpu_impl::CUvideoparserHandle(parser_info);

  // Init internal frame buffer
  // TODO(awolant): Check, if continuous buffer would be faster
  for (size_t i = 0; i < frame_buffer_.size(); ++i) {
    frame_buffer_[i].frame_.resize(FrameSize());
    frame_buffer_[i].pts_ = -1;
  }
}

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename, cudaStream_t stream, DALIImageType image_type)
    : FramesDecoderBase(filename, image_type),
      frame_buffer_(num_decode_surfaces_),
      stream_(stream) {
  is_valid_ = is_valid_ && SelectVideoStream();
  if (is_valid_) {
    InitGpuParser();
  }
}

FramesDecoderGpu::FramesDecoderGpu(const char *memory_file, size_t memory_file_size, std::string_view source_info,
                                   cudaStream_t stream, DALIImageType image_type)
    : FramesDecoderBase(memory_file, memory_file_size, source_info, image_type),
      frame_buffer_(num_decode_surfaces_),
      stream_(stream) {
  is_valid_ = is_valid_ && SelectVideoStream();
  if (is_valid_) {
    InitGpuParser();
  }
}

int FramesDecoderGpu::ProcessPictureDecode(CUVIDPICPARAMS *picture_params) {
  // Sending empty packet will call this callback.
  // If we want to flush the decoder, we do not need to do anything here
  if (flush_) {
    return 0;
  }

  CUDA_CALL(cuvidDecodePicture(nvdecode_state_->decoder, picture_params));
  CUVIDPARSERDISPINFO picture_display_info;
  memset(&picture_display_info, 0, sizeof(picture_display_info));
  picture_display_info.picture_index = picture_params->CurrPicIdx;
  picture_display_info.progressive_frame = !picture_params->field_pic_flag;
  picture_display_info.top_field_first = picture_params->bottom_field_flag ^ 1;
  HandlePictureDisplay(&picture_display_info);

  return 1;
}

int FramesDecoderGpu::HandlePictureDisplay(CUVIDPARSERDISPINFO *picture_display_info) {
  CUVIDPROCPARAMS videoProcessingParameters = {};
  videoProcessingParameters.progressive_frame = !picture_display_info->progressive_frame;
  videoProcessingParameters.second_field = 1;
  videoProcessingParameters.top_field_first = picture_display_info->top_field_first;
  videoProcessingParameters.unpaired_field = 0;
  videoProcessingParameters.output_stream = stream_;

  // If there are no piped pts, use the last pts available (unexpected extra frame)
  if (!piped_pts_.empty()) {
    current_pts_ = piped_pts_.front();
    piped_pts_.pop();
  }
  assert(current_pts_ != AV_NOPTS_VALUE);

  LOG_LINE << "HandlePictureDisplay-"
           << (picture_display_info->progressive_frame ?
                   "I" :
                   "NI")  // I=progressive frame, NI=interlaced
           << ": " << (current_copy_to_output_ ? "Read" : "Skip") << " frame, index "
           << next_frame_idx_ << ", timestamp " << std::setw(5) << current_pts_ << std::endl;

  // current_pts_ is pts of frame that came from the decoder
  // index_[NextFrameIdx()].pts is pts of the frame that we want to return
  // in this call to ReadNextFrame
  // If they are the same, we just return this frame
  // If not, we store it in the buffer for later

  void *frame_output = nullptr;
  if (HasIndex() && current_pts_ == index_[NextFrameIdx()].pts) {
    // Currently decoded frame is actually the one we want to display
    frame_returned_ = true;
    LOG_LINE << "Found frame with correct display timestamp " << current_pts_
              << " for index " << next_frame_idx_ << std::endl;
    if (current_copy_to_output_ == false) {
      return 1;
    }
    frame_output = current_frame_output_;
  } else {
    // Put currently decoded frame to the buffer for later
    auto &slot = FindEmptySlot();
    slot.pts_ = current_pts_;
    frame_output = slot.frame_.data();
  }

  CUdeviceptr frame = {};
  unsigned int pitch = 0;

  CUDA_CALL(cuvidMapVideoFrame(
    nvdecode_state_->decoder,
    picture_display_info->picture_index,
    &frame,
    &pitch,
    &videoProcessingParameters));

  LOG_LINE << "VideoColorSpaceConversion with conversion_type_="
           << static_cast<int>(conversion_type_) << " and dimensions " << Height() << "x" << Width()
           << std::endl;
  TYPE_SWITCH(dtype_, type2id, OutType, (uint8_t, float), (
    VideoColorSpaceConversion(
      reinterpret_cast<OutType *>(frame_output), Width() * 3,
      reinterpret_cast<const uint8_t *>(frame), static_cast<int>(pitch),
      Height(),
      Width(),
      conversion_type_,
      false,  // normalized_range_,
      stream_);
  ), DALI_FAIL(make_string("Unsupported type: ", dtype_)));

  // TODO(awolant): Alternative is to copy the data to a buffer
  // and then process it on the stream. Check, if this is faster, when
  // the benchmark is ready.
  CUDA_CALL(cudaStreamSynchronize(stream_));
  CUDA_CALL(cuvidUnmapVideoFrame(nvdecode_state_->decoder, frame));

  return 1;
}

void FramesDecoderGpu::SeekFrame(int frame_id) {
  // Will seek or reset decoder, otherwise we will only skip frames
  if (frame_id < next_frame_idx_) {
    SendLastPacket(true);
  }
  FramesDecoderBase::SeekFrame(frame_id);
}

bool FramesDecoderGpu::ReadNextFrameWithIndex(uint8_t *data) {
  bool copy_to_output = data != nullptr;

  // No more frames to read
  if (next_frame_idx_ >= NumFrames()) {
    next_frame_idx_ = -1;
    return false;
  }

  // Check if requested frame was buffered earlier
  assert(HasIndex());
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ != -1 && frame.pts_ == index_[next_frame_idx_].pts) {
      if (copy_to_output) {
        LOG_LINE << "Copying from frame buffer to frame_output" << std::endl;
        copyD2D(data, frame.frame_.data(), FrameSize(), stream_);
      }
      LOG_LINE << "Found buffered frame with pts=" << frame.pts_ << std::endl;

      frame.pts_ = -1;
      next_frame_idx_ = next_frame_idx_ == NumFrames() - 1 ? -1 : next_frame_idx_ + 1;
      return true;
    }
  }

  current_copy_to_output_ = copy_to_output;
  current_frame_output_ = data;

  while (next_frame_idx_ < NumFrames()) {
    int ret = av_read_frame(ctx_, packet_);
    auto packet = AVPacketScope(packet_, av_packet_unref);
    if (ret != 0) {
      LOG_LINE << "Hit EOF, sending last packet with " << piped_pts_.size()
               << " frames in pipeline, " << NumBufferedFrames() << " buffered frames, "
               << "next_frame_idx=" << next_frame_idx_ << " of " << NumFrames() << std::endl;
      break;
    }

    if (!SendFrameToParser()) {
      continue;
    } else {
      packet.release();  // ownership is transferred via av_bsf_send_packet
    }

    if (frame_returned_) {
      next_frame_idx_ = next_frame_idx_ == NumFrames() - 1 ? -1 : next_frame_idx_ + 1;
      return true;
    }
  }

  // At this point we've hit EOF but might have frames still in the pipeline
  DALI_ENFORCE(piped_pts_.size() >= 1);

  SendLastPacket();

  // Check if we got the frame we wanted during SendLastPacket
  if (frame_returned_) {
    next_frame_idx_ = next_frame_idx_ == NumFrames() - 1 ? -1 : next_frame_idx_ + 1;
    return true;
  }
  return true;
}

bool FramesDecoderGpu::SendFrameToParser() {
  if (packet_->stream_index != stream_id_) {
    return false;
  }

  // Store pts from current packet to indicate,
  // that this frame is in the decoder
  if (packet_->pts != AV_NOPTS_VALUE) {
    piped_pts_.push(packet_->pts);
  } else {
    piped_pts_.push(frame_index_if_no_pts_);
    frame_index_if_no_pts_++;
  }

  // Add header needed for NVDECODE to the packet
  if (filtered_packet_->data) {
    av_packet_unref(filtered_packet_);
  }

  LOG_LINE << "Sending packet to bitstream filter " << packet_->pts << std::endl;
  DALI_ENFORCE(av_bsf_send_packet(bsfc_, packet_) >= 0);
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
  LOG_LINE << "Sending packet to the nv decoder, next pts=" << piped_pts_.front() << std::endl;
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));
  LOG_LINE << "Packet sent to the nv decoder" << std::endl;
  current_pts_ = AV_NOPTS_VALUE;  // we are done with this packet. Next frame will come from a new packet
  return true;
}

bool FramesDecoderGpu::ReadNextFrameWithoutIndex(uint8_t *data) {
  current_copy_to_output_ = data != nullptr;
  current_frame_output_ = data;

  int64_t frame_to_return_index = -1;

  // Handle the case, when packet has more frames that we have empty spots
  // in the buffer.
  // If so, we need to return frame from the buffer before sending last packet.
  if (frame_index_if_no_pts_ != 0) {
    if (NumEmptySpots() < (piped_pts_.size())) {
      for (size_t i = 0; i < frame_buffer_.size(); ++i) {
        if (frame_buffer_[i].pts_ == NextFrameIdx()) {
          frame_to_return_index = i;
          break;
        }
      }
    }
  }

  // Initial fill of the buffer
  frame_returned_ = false;
  while (
    // as we may enlarge the buffer make sure to not decode more than num_decode_surfaces_ frames
    NumEmptySpots() > frame_buffer_.size() - num_decode_surfaces_ &&
    more_frames_to_decode_ &&
    !frame_returned_ &&
    frame_to_return_index == -1) {
    int ret = av_read_frame(ctx_, packet_);
    auto packet = AVPacketScope(packet_, av_packet_unref);
    if (ret == 0) {
      if (!SendFrameToParser()) {
        continue;
      }
    } else {
      // Handle the case, when last packet has more frames that we have empty spots
      // in the buffer.
      // If so, we need to return frame from the buffer before sending last packet.
      if (frame_index_if_no_pts_ != 0) {
        if (NumEmptySpots() < (piped_pts_.size())) {
          for (size_t i = 0; i < frame_buffer_.size(); ++i) {
            if (frame_buffer_[i].pts_ == NextFrameIdx()) {
              frame_to_return_index = i;
              break;
            }
          }
        }
      }

      if (frame_to_return_index == -1) {
        SendLastPacket();
        more_frames_to_decode_ = false;
      } else {
        break;
      }
    }
  }

  if (frame_to_return_index == -1) {
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
      if (frame_buffer_[i].pts_ != -1) {
        frame_to_return_index = i;
        break;
      }
    }

    for (size_t i = 1; i < frame_buffer_.size(); ++i) {
      if (frame_buffer_[i].pts_ != -1) {
        if (frame_buffer_[frame_to_return_index].pts_ > frame_buffer_[i].pts_) {
          frame_to_return_index = i;
        }
      }
    }
  }

  // This has to be separate if statement, because condition
  // might have changed in the previous one.
  if (frame_to_return_index == -1) {
    return true;
  }

  if (current_copy_to_output_) {
    assert(current_frame_output_ != nullptr);
    int64_t total_size = FrameSize();
    if (dtype_ == DALI_FLOAT) {
      total_size *= 4;
    }
    LOG_LINE << "Copying from frame buffer at index " << frame_to_return_index
             << " to frame_output at " << current_frame_output_ << " (frame_idx=" << next_frame_idx_
             << ")" << std::endl;
    copyD2D(
      static_cast<uint8_t *>(current_frame_output_),
      frame_buffer_[frame_to_return_index].frame_.data(),
      total_size,
      stream_);
  }

  LOG_LINE << (current_copy_to_output_ ? "Read" : "Skip") << " frame, index " << next_frame_idx_
           << ", timestamp " << std::setw(5) << frame_buffer_[frame_to_return_index].pts_
           << std::endl;

  ++next_frame_idx_;

  frame_buffer_[frame_to_return_index].pts_ = -1;

  if (IsBufferEmpty()) {
    next_frame_idx_ = -1;
  }

  return true;
}

bool FramesDecoderGpu::ReadNextFrame(uint8_t *data) {
  // No more frames in the file
  if (next_frame_idx_ == -1) {
    return false;
  }

  if (HasIndex()) {
    return ReadNextFrameWithIndex(data);
  } else {
    return ReadNextFrameWithoutIndex(data);
  }
}

unsigned int FramesDecoderGpu::NumBufferedFrames() const {
  unsigned int num_buffered = 0;
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ != -1) {
      num_buffered++;
    }
  }
  return num_buffered;
}

void FramesDecoderGpu::SendLastPacket(bool flush) {
  LOG_LINE << "SendLastPacket: flush=" << flush << " piped_pts_size=" << piped_pts_.size()
           << " buffered_frames=" << NumBufferedFrames() << " next_frame_idx=" << next_frame_idx_
           << std::endl;

  flush_ = flush;
  CUVIDSOURCEDATAPACKET *packet = &nvdecode_state_->packet;
  memset(packet, 0, sizeof(CUVIDSOURCEDATAPACKET));
  packet->payload = nullptr;
  packet->payload_size = 0;
  packet->flags = CUVID_PKT_ENDOFSTREAM;

  // Send end of stream and process any remaining frames
  CUDA_CALL(cuvidParseVideoData(nvdecode_state_->parser, packet));

  if (!flush) {
    // Process any remaining buffered frames in order
    bool found_frame;
    do {
      found_frame = false;
      for (auto &frame : frame_buffer_) {
        if (frame.pts_ != -1 && HasIndex() && frame.pts_ == index_[next_frame_idx_].pts) {
          LOG_LINE << "Processing remaining buffered frame pts=" << frame.pts_ << " for index "
                   << next_frame_idx_ << std::endl;
          frame.pts_ = -1;
          ++next_frame_idx_;
          found_frame = true;
          break;
        }
      }
    } while (found_frame && next_frame_idx_ < NumFrames());
  }

  flush_ = false;

  if (flush) {
    LOG_LINE << "Flushing decoder state" << std::endl;
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

  // in some cases we may decode more than one frame after receiving one packet
  // frame N-1 may require packet N and N-1, and in results after submitting packet N we
  // will get frame N-1 and N but the buffer may have space only for 1 frame
  std::vector<BufferedFrame> new_frame_buffer(frame_buffer_.size() + 1);
  for (size_t i = 0; i < frame_buffer_.size(); ++i) {
    new_frame_buffer[i] = std::move(frame_buffer_[i]);
  }
  frame_buffer_ = std::move(new_frame_buffer);
  auto &new_frame = frame_buffer_.back();
  new_frame.frame_.resize(FrameSize());
  new_frame.pts_ = -1;
  return new_frame;
}

bool FramesDecoderGpu::HasEmptySlot() const {
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == -1) {
      return true;
    }
  }
  return false;
}

bool FramesDecoderGpu::IsBufferEmpty() const {
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ != -1) {
      return false;
    }
  }

  return true;
}

unsigned int FramesDecoderGpu::NumEmptySpots() const {
  unsigned int num_empty = 0;
  for (auto &frame : frame_buffer_) {
    if (frame.pts_ == -1) {
      num_empty++;
    }
  }

  return num_empty;
}

void FramesDecoderGpu::Reset() {
  LOG_LINE << "Resetting decoder " << this << std::endl;
  Flush();
  FramesDecoderBase::Reset();
  LOG_LINE << "Reset decoder " << this << std::endl;
}

void FramesDecoderGpu::Flush() {
  LOG_LINE << "Flush" << std::endl;
  SendLastPacket(true);
  more_frames_to_decode_ = true;
  frame_index_if_no_pts_ = 0;
  LOG_LINE << "Flush done" << std::endl;
}

FramesDecoderGpu::~FramesDecoderGpu() {
  filtered_packet_.reset();
  bsfc_.reset();
}

bool FramesDecoderGpu::SupportsHevc() {
  CUVIDDECODECAPS decoder_caps = {};
  decoder_caps.eCodecType = cudaVideoCodec_HEVC;
  decoder_caps.eChromaFormat = cudaVideoChromaFormat_420;
  decoder_caps.nBitDepthMinus8 = 10 - 8;  // 10-bit HEVC
  CUDA_CALL(cuvidGetDecoderCaps(&decoder_caps));
  return decoder_caps.bIsSupported;
}

void FramesDecoderGpu::CopyFrame(uint8_t *dst, const uint8_t *src) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, FrameSize(), cudaMemcpyDeviceToDevice, stream_));
}

bool FramesDecoderGpu::SelectVideoStream(int stream_id) {
  if (!FramesDecoderBase::SelectVideoStream(stream_id)) {
    return false;
  }

  assert(stream_id_ >= 0 && stream_id_ < static_cast<int>(ctx_->nb_streams));
  assert(codec_params_);
  AVCodecID codec_id = codec_params_->codec_id;

  static constexpr std::array<AVCodecID, /*7*/ 6> codecs = {
    AVCodecID::AV_CODEC_ID_H264,
    AVCodecID::AV_CODEC_ID_HEVC,
    AVCodecID::AV_CODEC_ID_VP8,
    AVCodecID::AV_CODEC_ID_VP9,
    //AVCodecID::AV_CODEC_ID_MJPEG,  // TODO(janton): add support for MJPEG
    AVCodecID::AV_CODEC_ID_AV1,
    AVCodecID::AV_CODEC_ID_MPEG4,
  };
  if (std::find(codecs.begin(), codecs.end(), codec_id) == codecs.end()) {
    DALI_WARN(make_string("Codec ", codec_id, " (", avcodec_get_name(codec_id),
                          ") is not supported by the GPU variant of this operator."));
    return false;
  }

  CUVIDDECODECAPS decoder_caps = {};
  decoder_caps.eCodecType = GetCodecType(codec_id);
  decoder_caps.eChromaFormat = cudaVideoChromaFormat_420;
  decoder_caps.nBitDepthMinus8 = 0;
  CUDA_CALL(cuvidGetDecoderCaps(&decoder_caps));
  if (!decoder_caps.bIsSupported) {
    DALI_WARN(make_string("Codec ", avcodec_get_name(codec_id),
                          " is not supported by NVDEC on this platform."));
    return false;
  }

  codec_ctx_.reset(avcodec_alloc_context3(nullptr));
  DALI_ENFORCE(codec_ctx_, "Could not alloc av codec context");

  int ret = avcodec_parameters_to_context(codec_ctx_, codec_params_);
  DALI_ENFORCE(ret >= 0, make_string("Could not fill the codec based on parameters: ",
                                     av_error_string(ret)));
  return true;
}

}  // namespace dali
