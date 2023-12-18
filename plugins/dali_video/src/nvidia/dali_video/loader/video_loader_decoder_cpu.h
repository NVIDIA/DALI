// Copyright (c) 2021 - 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

#include "dali/operators/reader/loader/loader.h"
#include "dali_video/loader/frames_decoder.h"
#include "dali_video/loader/video_loader_decoder_base.h"


namespace dali_video {
using VideoSampleCpu = VideoSample<dali::CPUBackend>;

class VideoLoaderDecoderCpu
  : public dali::Loader<dali::CPUBackend, VideoSampleCpu, true>, VideoLoaderDecoderBase {
 public:
  explicit inline VideoLoaderDecoderCpu(const dali::OpSpec &spec) :
    dali::Loader<dali::CPUBackend, VideoSampleCpu, true>(spec),
    VideoLoaderDecoderBase(spec) { }

  void ReadSample(VideoSampleCpu &sample) override;

  void PrepareEmpty(VideoSampleCpu &sample) override;

  void Skip() override;

 protected:
  dali::Index SizeImpl() override;

  void PrepareMetadataImpl() override;

 private:
  void Reset(bool wrap_to_shard) override;

  std::vector<FramesDecoder> video_files_;
};

}  // namespace dali_video
