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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_VIDEO_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_VIDEO_LOADER_H_

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/pipeline/operators/reader/nvdecoder/nvdecoder.h"
#include "dali/pipeline/operators/reader/nvdecoder/sequencewrapper.h"
#include "dali/pipeline/operators/reader/nvdecoder/nvcuvid.h"

template<typename T>
using av_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;
template<typename T>
av_unique_ptr<T> make_unique_av(T* raw_ptr, void (*deleter)(T**)) {
    // libav resource free functions take the address of a pointer.
    return av_unique_ptr<T>(raw_ptr, [=] (T* data) {deleter(&data);});
}

namespace dali {
#ifdef HAVE_AVSTREAM_CODECPAR
auto codecpar(AVStream* stream) -> decltype(stream->codecpar);
#else
auto codecpar(AVStream* stream) -> decltype(stream->codec);
#endif

struct OpenFile {
  bool open = false;
  AVRational frame_base_;
  AVRational stream_base_;
  int frame_count_;

  int vid_stream_idx_;
  int last_frame_;

#ifdef HAVE_AVBSFCONTEXT
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


class VideoLoader : public Loader<GPUBackend, SequenceWrapper> {
 public:
  explicit inline VideoLoader(const OpSpec& spec,
    const std::vector<std::string>& filenames)
    : Loader<GPUBackend, SequenceWrapper>(spec),
      count_(spec.GetArgument<int>("count")),
      height_(0),
      width_(0),
      filenames_(filenames),
      // TODO(spanev) handle device_id != 0
      device_id_(0),
      codec_id_(0),
      done_(false) {
  }

  void init() {
    /* Required to use libavformat: Initialize libavformat and register all
     * the muxers, demuxers and protocols.
     */
    av_register_all();

    // TODO(spanev) Implem several files handling
    total_frame_count_ = get_or_open_file(filenames_[0]).frame_count_;

    // TODO(spanev) to change after deciding what we want
    int seq_per_epoch = total_frame_count_ / count_;
    frame_starts_.resize(seq_per_epoch);
    for (int i = 0; i < seq_per_epoch; ++i) {
      frame_starts_[i] = i * count_;
    }
    // TODO(spanev) change random engine
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(frame_starts_), std::end(frame_starts_), rng);
    current_frame_idx_ = 0;

    thread_file_reader_ = std::thread{&VideoLoader::read_file, this};
  }

  ~VideoLoader() noexcept {
    done_ = true;
    send_queue_.shutdown();
    if (vid_decoder_) {
      vid_decoder_->finish();
    }
    if (thread_file_reader_.joinable()) {
      try {
        thread_file_reader_.join();
      } catch (const std::system_error& e) {
        DALI_FAIL("System error joining thread: "
                   + e.what());
      }
    }
  }

  void PrepareEmpty(SequenceWrapper *tensor) override;
  void ReadSample(SequenceWrapper *tensor) override;
  Index Size() override;

  OpenFile& get_or_open_file(std::string filename);
  void seek(OpenFile& file, int frame);
  void read_file();
  void push_sequence_to_read(std::string filename, int frame, int count);
  void receive_frames(SequenceWrapper& sequence);
  std::pair<int, int> load_width_height(const std::string& filename);

 private:
  // Params
  int count_;
  int output_height_;
  int output_width_;
  int height_;
  int width_;
  std::vector<std::string> filenames_;

  int device_id_;
  int codec_id_;
  VideoLoaderStats stats_;

  std::unordered_map<std::string, OpenFile> open_files_;
  std::unique_ptr<NvDecoder> vid_decoder_;

  ThreadSafeQueue<FrameReq> send_queue_;

  std::thread thread_file_reader_;

  int total_frame_count_;
  std::vector<int> frame_starts_;
  unsigned current_frame_idx_;

  volatile bool done_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_VIDEO_LOADER_H_
