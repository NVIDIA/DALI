// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

namespace dali {
class VideoDecoderMixed : public VideoDecoderBase<MixedBackend, FramesDecoderGpu> {
 public:
  explicit VideoDecoderMixed(const OpSpec &spec) :
    VideoDecoderBase<MixedBackend, FramesDecoderGpu>(spec) {}
};

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderMixed, Mixed);

}  // namespace dali
