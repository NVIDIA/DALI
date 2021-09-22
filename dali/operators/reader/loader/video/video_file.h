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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_FILE_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_FILE_H_

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include "dali/core/common.h"

#include <vector>
#include <string>


namespace dali {
struct IndexEntry {
  int last_keyframe_id;
  int pts;
  bool is_keyframe;
  bool is_flush_frame;
};

struct AvState {
  AVFormatContext *ctx_ = nullptr;
  AVCodec *codec_ = nullptr;
  AVCodecParameters *codec_params_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVPacket *packet_ = nullptr;
  SwsContext  *sws_ctx_ = nullptr;
  int stream_id_ = -1;

  ~AvState() {
    sws_freeContext(sws_ctx_);
    avformat_close_input(&ctx_);
    avformat_free_context(ctx_);
    av_frame_free(&frame_);
    av_packet_free(&packet_);
    avcodec_free_context(&codec_ctx_);

    ctx_ = nullptr;
    codec_ = nullptr;
    codec_params_ = nullptr;
    codec_ctx_ = nullptr;
    frame_ = nullptr;
    packet_ = nullptr;
    sws_ctx_ = nullptr;
  }
};

class DLL_PUBLIC VideoFileCPU {
 public:
  VideoFileCPU(const std::string &filename);

  int64_t NumFrames() const {
    return index_.size();
  }

  int Width() const {
    return av_state_->codec_params_->width;
  }

  int Height() const {
    return av_state_->codec_params_->height;
  }

  int Channels() const {
      return channels_;
  }

  int FrameSize() const {
    return Channels() * Width() * Height();
  }

  void ReadNextFrame(uint8_t *data, bool copy_to_output = true);

  void SeekFrame(int frame_id);

 private:
  bool ReadRegularFrame(uint8_t *data, bool copy_to_output = true);

  bool ReadFlushFrame(uint8_t *data, bool copy_to_output = true);

  void CopyToOutput(uint8_t *data);

  void BuildIndex();

  void Reset();

  void InitAvState();

  void FindVideoStream();

  std::unique_ptr<AvState> av_state_;

  // SW parameters
  uint8_t *dest_[4] = {nullptr, nullptr, nullptr, nullptr};
  int dest_linesize_[4] = {0, 0, 0, 0};


  int channels_ = 3;
  bool flush_state_ = false;
  std::string filename_;
  std::vector<IndexEntry> index_;
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_FILE_H_
