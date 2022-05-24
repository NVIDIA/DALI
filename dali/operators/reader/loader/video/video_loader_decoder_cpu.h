// Copyright (c) 2021 - 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_CPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_CPU_H_

#include <string>
#include <vector>

#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/video_loader_decoder_base.h"


namespace dali {
using VideoSampleCpu = VideoSample<CPUBackend>;

class VideoLoaderDecoderCpu : public Loader<CPUBackend, VideoSampleCpu>, VideoLoaderDecoderBase {
 public:
  explicit inline VideoLoaderDecoderCpu(const OpSpec &spec) :
    Loader<CPUBackend, VideoSampleCpu>(spec),
    VideoLoaderDecoderBase(spec) { }

  void ReadSample(VideoSampleCpu &sample) override;

  void PrepareEmpty(VideoSampleCpu &sample) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override;

 private:
  void Reset(bool wrap_to_shard) override;

  std::vector<FramesDecoder> video_files_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_CPU_H_
