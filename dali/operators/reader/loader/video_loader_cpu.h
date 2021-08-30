// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_CPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_CPU_H_

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#include <string>
#include <vector>
#include <algorithm>

#include "dali/operators/reader/loader/loader.h"

namespace dali {
class VideoFileCPU {
 public:
  VideoFileCPU(std::string &filename);

  int64_t nb_frames() const {
    return ctx_->streams[stream_id_]->nb_frames;
  }

  int width() const {
    return codec_params_->width;
  }

  int height() const {
    return codec_params_->height;
  }

  void read_next_frame(uint8_t *data) {

  }

  AVFormatContext *ctx_ = nullptr;
  AVCodec *codec_ = nullptr;
  AVCodecParameters *codec_params_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  int stream_id_ = -1;

  int width_;
  int height_;
  int channels_ = 3;

  int read_frames = 0;
};

class VideoSampleSpan {
 public:
  explicit VideoSampleSpan(int start, int end) : 
    start_(start), end_(end) {}

  int start_ = -1;
  int end_ = -1;
};


class VideoLoaderCPU : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit inline VideoLoaderCPU(const OpSpec &spec) : 
    Loader<CPUBackend, Tensor<CPUBackend>>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    sequence_len_(spec.GetArgument<int>("sequence_length")) {
  }

  void ReadSample(Tensor<CPUBackend> &sample) override;

protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override;

private:
  void Reset(bool wrap_to_shard) override;

  std::vector<std::string> filenames_;
  std::vector<VideoFileCPU> video_files_;
  std::vector<VideoSampleSpan> sample_spans_;

  Index current_index_ = 0;

  int sequence_len_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_CPU_H_
