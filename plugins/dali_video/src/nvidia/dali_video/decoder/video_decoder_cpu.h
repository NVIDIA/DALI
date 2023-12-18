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

#pragma once

#include <vector>
#include <memory>
#include "dali_video/decoder/video_decoder_base.h"
#include "dali_video/loader/frames_decoder.h"
#include "dali/pipeline/operator/operator.h"

namespace dali_video {

class VideoDecoderCpu
        : public dali::Operator<dali::CPUBackend>, public VideoDecoderBase<dali::CPUBackend, FramesDecoder> {
  using VideoDecoderBase::DecodeSample;

 public:
  explicit VideoDecoderCpu(const dali::OpSpec &spec) : dali::Operator<dali::CPUBackend>(spec) {}


  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc, const dali::Workspace &ws) override;

  void RunImpl(dali::Workspace &ws) override;
};

}   // namespace dali_video