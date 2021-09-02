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
#include <libswscale/swscale.h>
}

#include "dali/operators/reader/loader/loader.h"

namespace dali {
struct IndexEntry {
  bool is_keyframe;
  int last_keyframe_id;
  int pts;
};

class VideoFileCPU {
 public:
  VideoFileCPU(std::string &filename);

  int64_t NumFrames() const {
    return num_frames_;
  }

  int Width() const {
    return codec_params_->width;
  }

  int Height() const {
    return codec_params_->height;
  }

  int FrameSize() const {
    return channels_ * Width() * Height();
  }

  // Reads next frame of the video and wraps at the end
  void ReadNextFrame(uint8_t *data);

  AVFormatContext *ctx_ = nullptr;
  AVCodec *codec_ = nullptr;
  AVCodecParameters *codec_params_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVPacket *packet_ = nullptr;
  SwsContext  *sws_ctx_ = nullptr;
  int stream_id_ = -1;

  int channels_ = 3;
  int num_frames_ = 0;

  std::vector<IndexEntry> index_;

  uint8_t *dest_[4] = {nullptr, nullptr, nullptr, nullptr};
  int dest_linesize_[4] = {0, 0, 0, 0};

 private:
  bool ReadRegularFrame(uint8_t *data, bool copy_to_output = true);

  bool ReadFlushFrame(uint8_t *data, bool copy_to_output = true);

  void SeekFrame(int frame_id);

  void CopyToOutput(uint8_t *data);

  void BuildIndex();

  bool flush_state_ = false;
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
