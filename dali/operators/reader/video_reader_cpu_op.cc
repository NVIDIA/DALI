// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/video_reader_cpu_op.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include "dali/test/cv_mat_utils.h"

#include <string>
#include <vector>
#include <algorithm>

namespace dali {

class VideoFileCPU {
 public:
  VideoFileCPU(std::string &filename) {
    ctx_ = avformat_alloc_context();
    avformat_open_input(&ctx_, filename.c_str(), nullptr, nullptr);

    for (size_t i = 0; i < ctx_->nb_streams; ++i) {
      auto stream = ctx_->streams[i];
      codec_params_ = ctx_->streams[i]->codecpar;
      codec_ = avcodec_find_decoder(codec_params_->codec_id);

      if (codec_->type == AVMEDIA_TYPE_VIDEO) {
          stream_id_ = i;
          break;
      }
    }

    codec_ctx_ = avcodec_alloc_context3(codec_);
    avcodec_parameters_to_context(codec_ctx_, codec_params_);
    avcodec_open2(codec_ctx_, codec_, nullptr);
  }

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

  void ReadSample(Tensor<CPUBackend> &sample) override {
    auto &sample_span = sample_spans_[current_index_];
    auto &video_file = video_files_[0];

    ++current_index_;
    // char str[4];
    // snprintf(str, 4, "%03d", current_index_-1);
    // string path = "/home/awolant/Downloads/frames/" + string(str) + ".png";
    MoveToNextShard(current_index_);

    sample.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
    sample.Resize(
      TensorShape<4>{sequence_len_, video_file.width(), video_file.height(), video_file.channels_});

    //TODO: przesunąć na kolejną ramkę z sekwencji (sekwencja jest dłuższa niż jedna ramka)
    auto data = sample.mutable_data<uint8_t>();

    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();

    int ret = -1;
    int frames_count = 0;
    while ((ret = av_read_frame(video_file.ctx_, packet)) >= 0) {
      if (packet->stream_index != video_file.stream_id_) {
        continue;
      }

      ret = avcodec_send_packet(video_file.codec_ctx_, packet);
      ret = avcodec_receive_frame(video_file.codec_ctx_, frame);

      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        continue;
      }

      

      SwsContext  *sws_ctx = sws_getContext(
        frame->width, 
        frame->height, 
        video_file.codec_ctx_->pix_fmt, 
        frame->width, 
        frame->height, 
        AV_PIX_FMT_RGB24, 
        SWS_BILINEAR, 
        nullptr, 
        nullptr, 
        nullptr);

      uint8_t *dest[4] = {data, nullptr, nullptr, nullptr};
      int dest_linesize[4] = {frame->width * 3, 0, 0, 0};
      sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

      ++frames_count;

      // TensorView<StorageCPU, uint8_t> tv(data, TensorShape<3>{720, 1280, 3});
      // testing::SaveImage(path.c_str(), tv);

      if (frames_count == sequence_len_) {
        break;
      }
    }

    if (frames_count == sequence_len_) {
      return;
    }

    ret = avcodec_send_packet(video_file.codec_ctx_, nullptr);

    ret = avcodec_receive_frame(video_file.codec_ctx_, frame);

    if (ret < 0) {
      return;
    }

    SwsContext  *sws_ctx = sws_getContext(
      frame->width, 
      frame->height, 
      video_file.codec_ctx_->pix_fmt, 
      frame->width, 
      frame->height, 
      AV_PIX_FMT_RGB24, 
      SWS_BILINEAR, 
      nullptr, 
      nullptr, 
      nullptr);

    uint8_t *dest[4] = {data, nullptr, nullptr, nullptr};
    int dest_linesize[4] = {frame->width * 3, 0, 0, 0};
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

    ++frames_count;

    // TensorView<StorageCPU, uint8_t> tv(data, TensorShape<3>{720, 1280, 3});
    // testing::SaveImage(path.c_str(), tv);

    if (frames_count == sequence_len_) {
      return;
    }
  }

protected:
  Index SizeImpl() override {
    return sample_spans_.size();
  }

  void PrepareMetadataImpl() override {
    for (auto &filename : filenames_) {
      video_files_.push_back(VideoFileCPU(filename));
    }

    for (int start = 0; start + sequence_len_ <= video_files_[0].nb_frames(); start += sequence_len_) {
      sample_spans_.push_back(VideoSampleSpan(start, start + sequence_len_));
    }
  }

private:
  void Reset(bool wrap_to_shard) override {
    current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
  }


  std::vector<std::string> filenames_;
  std::vector<VideoFileCPU> video_files_;
  std::vector<VideoSampleSpan> sample_spans_;

  Index current_index_ = 0;

  int sequence_len_;
};

VideoReaderCPU::VideoReaderCPU(const OpSpec &spec)
    : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
      loader_ = InitLoader<VideoLoaderCPU>(spec);
}

void VideoReaderCPU::RunImpl(SampleWorkspace &ws) {
  const auto &video_sample = GetSample(ws.data_idx());
  auto &video_output = ws.Output<CPUBackend>(0);

  video_output.Copy(video_sample, 0);
}

}  // namespace dali
