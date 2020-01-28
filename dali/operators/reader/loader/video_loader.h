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
#include <list>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/nvdecoder/nvdecoder.h"
#include "dali/operators/reader/nvdecoder/sequencewrapper.h"

template<typename T>
using av_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;
template<typename T>
av_unique_ptr<T> make_unique_av(T* raw_ptr, void (*deleter)(T**)) {
    // libav resource free functions take the address of a pointer.
    return av_unique_ptr<T>(raw_ptr, [=] (T* data) {deleter(&data);});
}

using nvdecDriverHandle = std::unique_ptr<std::remove_pointer<DLLDRIVER>::type,
                                          std::function< void(DLLDRIVER) >>;

namespace dali {
#if HAVE_AVSTREAM_CODECPAR
auto codecpar(AVStream* stream) -> decltype(stream->codecpar);
#else
auto codecpar(AVStream* stream) -> decltype(stream->codec);
#endif

struct file_meta {
  std::string video_file;
  int label;
  float start_time;
  float end_time;
  bool operator< (const file_meta& right) {
    return video_file < right.video_file;
  }
};

namespace filesystem {

std::vector<dali::file_meta> get_file_label_pair(const std::string& path,
    const std::vector<std::string>& filenames, const std::string& file_list);

}  // namespace filesystem

struct VideoFileDesc {
  FILE *file_stream = nullptr;
  uint64_t file_position = 0;
  std::string filename;
  ~VideoFileDesc() {
    if (file_stream) {
      fclose(file_stream);
      file_stream = nullptr;
    }
  }
};

struct VideoFile {
  AVRational frame_base_;
  AVRational stream_base_;
  int64_t start_time_ = 0;
  int frame_count_ = -1;

  int vid_stream_idx_ = -1;
  int last_frame_ = -1;

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
  AVCodecContext* codec = nullptr;
#endif
  av_unique_ptr<AVFormatContext> fmt_ctx_;

  VideoFileDesc file_desc;
  bool empty() const noexcept { return file_desc.filename.empty(); }
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
      file_list_frame_num_(spec.GetArgument<bool>("file_list_frame_num")),
      stats_({0, 0, 0, 0, 0}),
      current_frame_idx_(-1),
      stop_(false) {
    if (step_ < 0)
      step_ = count_ * stride_;

    file_info_ = filesystem::get_file_label_pair(file_root_, filenames_, file_list_);
    DALI_ENFORCE(!file_info_.empty(), "No files were read.");

    lib_handle_ = nvdecDriverHandle(cuvidInitChecked(0), cuvidDeinit);

    DALI_ENFORCE(lib_handle_,
      "Failed to load libnvcuvid.so, needed by the VideoReader operator. "
      "If you are running in a Docker container, please refer "
      "to https://github.com/NVIDIA/nvidia-docker/wiki/Usage");
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

  VideoFile& get_or_open_file(const std::string &filename);
  void seek(VideoFile& file, int frame);
  void read_file();
  void push_sequence_to_read(std::string filename, int frame, int count);
  void receive_frames(SequenceWrapper& sequence);

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    int total_count = 1 + (count_ - 1) * stride_;

    for (size_t i = 0; i < file_info_.size(); ++i) {
      const auto& file = get_or_open_file(file_info_[i].video_file);
      const auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
      int frame_count = file.frame_count_;

      int start_frame = 0;
      int end_frame = file.frame_count_;
      float start = file_info_[i].start_time;
      float end = file_info_[i].end_time;
      if (start != -1 && end != -1) {
        if (file_list_frame_num_) {
          start_frame = start;
          end_frame = end;
          DALI_ENFORCE(end_frame <= file.frame_count_, "End frame number is greater than "
              "total number of frames for file " + file_info_[i].video_file);
        } else {
          auto frame_rate = av_inv_q(file.frame_base_);
          start_frame = static_cast<int>(std::ceil(start * av_q2d(frame_rate)));
          end_frame = static_cast<int>(std::floor(end * av_q2d(frame_rate)));

          DALI_ENFORCE(end_frame <= file.frame_count_, "End time is greater than video duration "
                       "for file " + file_info_[i].video_file);
        }
      }

      for (int s = start_frame; s < end_frame && s + total_count <= end_frame; s += step_) {
        frame_starts_.emplace_back(sequence_meta{i, s, file_info_[i].label,
                                   codecpar(stream)->height, codecpar(stream)->width});
      }
    }


    const auto& file = get_or_open_file(file_info_[0].video_file);
    auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];

    vid_decoder_ = std::make_unique<NvDecoder>(device_id_,
                                               codecpar(stream),
                                               image_type_,
                                               dtype_,
                                               normalized_,
                                               ALIGN16(max_height_),
                                               ALIGN16(max_width_),
                                               additional_decode_surfaces_);

    if (shuffle_) {
      // TODO(spanev) decide of a policy for multi-gpu here and SequenceLoader
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
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
  bool file_list_frame_num_;
  VideoLoaderStats stats_;

  std::unordered_map<std::string, VideoFile> open_files_;
  std::string last_opened_;
  nvdecDriverHandle lib_handle_;
  std::unique_ptr<NvDecoder> vid_decoder_;

  ThreadSafeQueue<FrameReq> send_queue_;

  std::thread thread_file_reader_;

  std::vector<struct sequence_meta> frame_starts_;
  Index current_frame_idx_;

  volatile bool stop_;
  std::vector<file_meta> file_info_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_LOADER_H_
