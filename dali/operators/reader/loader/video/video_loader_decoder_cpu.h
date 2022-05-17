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

namespace dali {
class VideoSampleDesc {
 public:
  explicit VideoSampleDesc(int start, int end, int stride, int video_idx) :
    start_(start), end_(end), stride_(stride), video_idx_(video_idx) {}

  int start_ = -1;
  int end_ = -1;
  int stride_ = -1;
  int video_idx_ = -1;
};

template <typename Backend>
class VideoSample {
 public:
  Tensor<Backend> data_;
  int label_;
};

using VideoSampleCpu = VideoSample<CPUBackend>;

class VideoLoaderDecoderCpu : public Loader<CPUBackend, VideoSampleCpu> {
 public:
  explicit inline VideoLoaderDecoderCpu(const OpSpec &spec) :
    Loader<CPUBackend, VideoSampleCpu>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    sequence_len_(spec.GetArgument<int>("sequence_length")),
    stride_(spec.GetArgument<int>("stride")),
    step_(spec.GetArgument<int>("step")) {
    if (step_ <= 0) {
      step_ = stride_ * sequence_len_;
    }
    has_labels_ = spec.TryGetRepeatedArgument(labels_, "labels");
    DALI_ENFORCE(!has_labels_ || labels_.size() == filenames_.size(),
                 make_string("Number of provided files and lables should match. Provided ",
                             filenames_.size(), " files and ", labels_.size(),  " labels."));
  }

  void ReadSample(VideoSampleCpu &sample) override;

  void PrepareEmpty(VideoSampleCpu &sample) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override;

 private:
  void Reset(bool wrap_to_shard) override;

  std::vector<std::string> filenames_;
  std::vector<int> labels_;
  bool has_labels_ = false;
  std::vector<FramesDecoder> video_files_;
  std::vector<VideoSampleDesc> sample_spans_;

  Index current_index_ = 0;

  int sequence_len_;
  int stride_;
  int step_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_CPU_H_
