// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_VIDEO_DECODER_MIXED_H_
#define PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_VIDEO_DECODER_MIXED_H_

#include <vector>
#include <memory>
#include "dali/pipeline/operator/operator.h"
#include "VideoCodecSDKUtils/helper_classes/Utils/FFmpegDemuxer.h"
#include "VideoCodecSDKUtils/helper_classes/NvCodec/NvDecoder/NvDecoder.h"
#include "VideoCodecSDKUtils/helper_classes/Utils/NvCodecUtils.h"


namespace dali_video {

class VideoDecoderMixed : public dali::Operator<dali::MixedBackend> {
 public:
  explicit VideoDecoderMixed(const dali::OpSpec &spec)
    : Operator<dali::MixedBackend>(spec)
    , device_id_(spec.GetArgument<int>("device_id")) {
      if (spec.HasArgument("end_frame")) {
        end_frame_ = spec.GetArgument<int>("end_frame");
      } else {
        DALI_FAIL("Currently, `end_frame` argument is required.");
      }
    }

  void ValidateInput(const dali::Workspace &ws) {
    const auto &input = ws.Input<dali::CPUBackend>(0);
    DALI_ENFORCE(input.type() == dali::DALI_UINT8,
                 "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1,
                 "Input buffer must be 1-dimensional.");
    for (int64_t i = 0; i < input.num_samples(); ++i) {
      DALI_ENFORCE(input[i].shape().num_elements() > 0,
                   dali::make_string("Incorrect sample at position: ", i, ". ",
                                     "Video decoder does not support empty input samples."));
    }
  }

  void RunImpl(dali::Workspace &ws) override;

  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;

 private:
  int device_id_;
  int end_frame_ = -1;

  struct SampleCtx {
    std::unique_ptr<FFmpegDemuxer::DataProvider> data_provider_;
    std::unique_ptr<FFmpegDemuxer> demuxer_;
    std::unique_ptr<NvDecoder> decoder_;
    std::shared_ptr<PacketData> current_packet_;
  };
  std::vector<SampleCtx> samples_;
};

}  // namespace dali_video

#endif  // PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_VIDEO_DECODER_MIXED_H_
