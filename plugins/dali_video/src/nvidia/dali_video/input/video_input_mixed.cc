// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali_video/input/video_input.h"
#include <memory>

namespace dali_video {


template<>
void VideoInput<dali::MixedBackend, FramesDecoderGpu>::CreateDecoder(const dali::Workspace &ws) {
  auto sample = encoded_video_[0];
  auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
  size_t size = sample.shape().num_elements();
  this->frames_decoders_[0] = std::make_unique<FramesDecoderGpu>(data, size, ws.stream(),
                                                                       false);
}


class VideoInputMixed : public VideoInput<dali::MixedBackend> {
  /*
   * This awkward class originates from an API inconsistency between
   * Operator<dali::CPUBackend> and Operator<MixedBackend>. Operator<dali::CPUBackend> has a `RunImpl` function
   * to be overriden, while Operator<MixedBackend> has `Run` function to be overriden.
   * Can't sort it out using SFINAE, since these are virtual functions.
   */
 public:
  explicit VideoInputMixed(const dali::OpSpec &spec) : VideoInput<dali::MixedBackend>(spec) {}
  void Run(dali::Workspace &ws) override { VideoInputRunImpl(ws); }
};

DALI_REGISTER_OPERATOR(plugin__video__decoders__input__Video, VideoInputMixed, dali::Mixed);

}  // namespace dali_video
