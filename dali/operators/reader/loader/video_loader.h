// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_H_

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
}

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/nvdecoder/nvdecoder.h"
#include "dali/operators/reader/nvdecoder/sequencewrapper.h"
#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"

template<typename T>
using av_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;
template<typename T>
av_unique_ptr<T> make_unique_av(T* raw_ptr, void (*deleter)(T**)) {
    // libav resource free functions take the address of a pointer.
    return av_unique_ptr<T>(raw_ptr, [=] (T* data) {deleter(&data);});
}

namespace dali {
#if HAVE_AVSTREAM_CODECPAR
auto codecpar(AVStream* stream) -> decltype(stream->codecpar);
#else
auto codecpar(AVStream* stream) -> decltype(stream->codec);
#endif

namespace filesystem {

std::vector<std::pair<std::string, int>> get_file_label_pair(const std::string& path,
    const std::vector<std::string>& filenames, const std::string& file_list);

}  // namespace filesystem

struct OpenFile {
  bool open = false;
  AVRational frame_base_;
  AVRational stream_base_;
  int64_t start_time_;
  int frame_count_;

  int vid_stream_idx_;
  int last_frame_;

#if HAVE_AVBSFCONTEXT
  av_unique_ptr<AVBSFContext> bsf_ctx_;
#else
  struct BSFDeleter {
    void operator()(AVBitStreamFilterContext* bsf) {
      av_bitstream_filter_close(bsf);
    }
  };
  using bsf_ptr = std::unique_ptr<AVBitStreamFilterContext, BSFDeleter>;
  bsf_ptr bsf_ctx_;
  AVCodecContext* codec;
#endif
  av_unique_ptr<AVFormatContext> fmt_ctx_;
};

struct VideoLoaderStats {
  /** Total number of bytes read from disk */
  uint64_t bytes_read;

  /** Number of compressed packets read from disk */
  uint64_t packets_read;

  /** Total number of bytes sent to NVDEC for decoding, can be
   *  different from bytes_read when seeking is a bit off or if
   *  there are extra streams in the video file. */
  uint64_t bytes_decoded;

  /** Total number of packets sent to NVDEC for decoding, see bytes_decoded */
  uint64_t packets_decoded;

  /** Total number of frames actually used. This is usually less
   *  than packets_decoded because decoding must happen key frame to
   *  key frame and output sequences often span key frame sequences,
   *  requiring more frames to be decoded than are actually used in
   *  the output. */
  uint64_t frames_used;
};

struct sequence_meta {
  size_t filename_idx;
  int frame_idx;
  int label;
  int height;
  int width;
};


class VideoLoader : public Loader<GPUBackend, SequenceWrapper> {
 public:
  explicit inline VideoLoader(const OpSpec& spec,
    const std::vector<std::string>& filenames)
    : Loader<GPUBackend, SequenceWrapper>(spec),
      file_root_(spec.GetArgument<std::string>("file_root")),
      file_list_(spec.GetArgument<std::string>("file_list")),
      count_(spec.GetArgument<int>("sequence_length")),
      step_(spec.GetArgument<int>("step")),
      stride_(spec.GetArgument<int>("stride")),
      max_height_(0),
      max_width_(0),
      additional_decode_surfaces_(spec.GetArgument<int>("additional_decode_surfaces")),
      image_type_(spec.GetArgument<DALIImageType>("image_type")),
      dtype_(spec.GetArgument<DALIDataType>("dtype")),
      normalized_(spec.GetArgument<bool>("normalized")),
      filenames_(filenames),
      device_id_(spec.GetArgument<int>("device_id")),
      codec_id_(0),
      skip_vfr_check_(spec.GetArgument<bool>("skip_vfr_check")),
      stats_({0, 0, 0, 0, 0}),
      stop_(false) {
    if (step_ < 0)
      step_ = count_ * stride_;

    file_label_pair_ = filesystem::get_file_label_pair(file_root_, filenames_,
                                                       file_list_);
    DALI_ENFORCE(!file_label_pair_.empty(), "No files were read.");

    DALI_ENFORCE(cuvidInitChecked(0),
     "Failed to load libnvcuvid.so, needed by the VideoReader operator. "
     "If you are running in a Docker container, please refer "
     "to https://github.com/NVIDIA/nvidia-docker/wiki/Usage");
    /* Required to use libavformat: Initialize libavformat and register all
     * the muxers, demuxers and protocols.
     */
  }

  ~VideoLoader() noexcept override {
    stop_ = true;
    send_queue_.shutdown();
    if (vid_decoder_) {
      vid_decoder_->finish();
    }
    if (thread_file_reader_.joinable()) {
      try {
        thread_file_reader_.join();
      } catch (const std::system_error& e) {
        // We should not throw here
      }
    }
  }

  void PrepareEmpty(SequenceWrapper &tensor) override;
  void ReadSample(SequenceWrapper &tensor) override;

  OpenFile& get_or_open_file(const std::string &filename);
  void seek(OpenFile& file, int frame);
  void read_file();
  void push_sequence_to_read(std::string filename, int frame, int count);
  void receive_frames(SequenceWrapper& sequence);

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    int total_count = 1 + (count_ - 1) * stride_;

    for (size_t i = 0; i < file_label_pair_.size(); ++i) {
      const auto& file = get_or_open_file(file_label_pair_[i].first);
      const auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
      int frame_count = file.frame_count_;

      for (int s = 0; s < frame_count && s + total_count <= frame_count; s += step_) {
        frame_starts_.emplace_back(sequence_meta{i, s, file_label_pair_[i].second,
                                   codecpar(stream)->height, codecpar(stream)->width});
      }
    }


    const auto& file = get_or_open_file(file_label_pair_[0].first);
    auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];

    vid_decoder_ = std::unique_ptr<NvDecoder>{
        new NvDecoder(device_id_,
                      codecpar(stream),
                      stream->time_base,
                      image_type_,
                      dtype_,
                      normalized_,
                      ALIGN16(max_height_),
                      ALIGN16(max_width_),
                      additional_decode_surfaces_)};

    if (shuffle_) {
      // TODO(spanev) decide of a policy for multi-gpu here and SequenceLoader
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(std::begin(frame_starts_), std::end(frame_starts_), g);
    }

    Reset(true);

    thread_file_reader_ = std::thread{&VideoLoader::read_file, this};
  }

 private:
  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_frame_idx_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_frame_idx_ = 0;
    }
  }
  // Params
  std::string file_root_;
  std::string file_list_;
  int count_;
  int step_;
  int stride_;
  int max_height_;
  int max_width_;
  int additional_decode_surfaces_;
  static constexpr int channels_ = 3;
  DALIImageType image_type_;
  DALIDataType dtype_;
  bool normalized_;

  std::vector<std::string> filenames_;

  int device_id_;
  int codec_id_;
  bool skip_vfr_check_;
  VideoLoaderStats stats_;

  std::unordered_map<std::string, OpenFile> open_files_;
  std::unique_ptr<NvDecoder> vid_decoder_;

  ThreadSafeQueue<FrameReq> send_queue_;

  std::thread thread_file_reader_;

  std::vector<struct sequence_meta> frame_starts_;
  Index current_frame_idx_;

  volatile bool stop_;
  std::vector<std::pair<std::string, int>> file_label_pair_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_H_
