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

#include "dali/operators/reader/loader/video/frames_decoder_cpu.h"
#include <algorithm>
#include <string>
#include "dali/core/error_handling.h"

namespace dali {

FramesDecoderCpu::FramesDecoderCpu(const std::string &filename, bool build_index)
    : FramesDecoderBase(filename, build_index, true) {
  is_valid_ = is_valid_ && CanDecode(av_state_->codec_params_->codec_id);
}

FramesDecoderCpu::FramesDecoderCpu(const char *memory_file, size_t memory_file_size,
                                   bool build_index, int num_frames,
                                   std::string_view source_info)
    : FramesDecoderBase(memory_file, memory_file_size, build_index, true, num_frames,
                        source_info) {
  is_valid_ = is_valid_ && CanDecode(av_state_->codec_params_->codec_id);
}

bool FramesDecoderCpu::CanDecode(AVCodecID codec_id) const {
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

  void *iter = NULL;
  const AVCodec *codec = NULL;
  while ((codec = av_codec_iterate(&iter))) {
    if (codec->id == codec_id && av_codec_is_decoder(codec)) {
      return true;
    }
  }
  DALI_WARN(
      make_string("Codec ", codec_id, " (", avcodec_get_name(codec_id),
                  ") is not supported by the libavcodec version provided by DALI, and therefore "
                  "cannot be decoded on the CPU."));
  return false;
}

}  // namespace dali
