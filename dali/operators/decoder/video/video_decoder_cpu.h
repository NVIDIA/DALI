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

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_CPU_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_CPU_H_

#include <vector>
#include <memory>
#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

class VideoDecoderCpu
        : public Operator<CPUBackend>, public VideoDecoderBase<CPUBackend, FramesDecoder> {
  using VideoDecoderBase::DecodeSample;

 public:
  explicit VideoDecoderCpu(const OpSpec &spec) : Operator<CPUBackend>(spec) {}


  bool CanInferOutputs() const override {
    return true;
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;
};

}   // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_CPU_H_
