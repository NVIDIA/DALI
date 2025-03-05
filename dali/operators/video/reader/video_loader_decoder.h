// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_BASE_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_BASE_H_


#include <string>
#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/video/frames_decoder_base.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/frames_decoder_cpu.h"

namespace dali {

struct VideoSampleDesc {
  VideoSampleDesc(std::string filename = {}, int label = -1, int start = -1, int end = -1,
                  int stride = -1)
      : filename_(filename), label_(label), start_(start), end_(end), stride_(stride) {}
  std::string filename_;
  int label_ = -1;
  int start_ = -1;
  int end_ = -1;
  int stride_ = -1;
};

template <typename Backend>
struct VideoSample : public VideoSampleDesc {
  VideoSample(std::string filename = {}, int label = -1, int start = -1, int end = -1,
              int stride = -1)
      : VideoSampleDesc{filename, label, start, end, stride} {}

  VideoSample(const VideoSampleDesc &other) noexcept
      : VideoSampleDesc(other) {}

  // to be filled by Prefetch
  Tensor<Backend> data_;
  int64_t start_timestamp_ = -1;
};

template <typename Backend, typename FramesDecoderImpl, typename Sample = VideoSample<Backend>>
class VideoLoaderDecoder : public Loader<Backend, Sample, true> {
 public:
  explicit inline VideoLoaderDecoder(const OpSpec &spec):
    Loader<Backend, Sample, true>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    sequence_len_(spec.GetArgument<int>("sequence_length")),
    stride_(spec.GetArgument<int>("stride")),
    step_(spec.GetArgument<int>("step")) {
    has_labels_ = spec.TryGetRepeatedArgument(labels_, "labels");
    DALI_ENFORCE(
        !has_labels_ || labels_.size() == filenames_.size(),
        make_string(
            "Number of provided files and labels should match. Provided ",
            filenames_.size(), " files and ", labels_.size(), " labels."));
    if (step_ <= 0) {
      step_ = stride_ * sequence_len_;
    }
  }

  void PrepareEmpty(Sample &sample) {
    sample = Sample();
  }

  void ReadSample(Sample &sample) override {
    sample = Sample(samples_[current_index_]);
    MoveToNextShard(++current_index_);
  }

  void Skip() override {
    MoveToNextShard(++current_index_);
  }

  Index SizeImpl() override {
    return samples_.size();
  }

  void PrepareMetadataImpl() override {
    std::unique_ptr<FramesDecoderImpl> decoder;
    for (size_t i = 0; i < filenames_.size(); ++i) {
      const auto &filename = filenames_[i];
      int label = has_labels_ ? labels_[i] : -1;
      decoder = std::make_unique<FramesDecoderImpl>(filename, true);
      if (!decoder->IsValid()) {
        LOG_LINE << "Invalid video file: " << filename << std::endl;
        continue;
      }
      int64_t num_frames = decoder->NumFrames();
      for (int start = 0; start + stride_ * sequence_len_ <= num_frames;
           start += step_) {
        LOG_LINE << "Sample #" << samples_.size() << ": " << filename << " " << label << " "
                 << start << ".." << start + stride_ * sequence_len_ << std::endl;
        samples_.emplace_back(filename, label, start, start + stride_ * sequence_len_, stride_);
      }
    }

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(std::begin(samples_), std::end(samples_), g);
    }

    // set the initial index for each shard
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    current_index_ = wrap_to_shard ? start_index(virtual_shard_id_, num_shards_, SizeImpl()) : 0;
  }

 protected:
  using Base = Loader<Backend, Sample, true>;
  using Base::shard_id_;
  using Base::virtual_shard_id_;
  using Base::num_shards_;
  using Base::stick_to_shard_;
  using Base::shuffle_;
  using Base::dont_use_mmap_;
  using Base::initial_buffer_fill_;
  using Base::copy_read_data_;
  using Base::read_ahead_;
  using Base::IsCheckpointingEnabled;
  using Base::PrepareEmptyTensor;
  using Base::MoveToNextShard;
  using Base::ShouldSkipImage;

  std::vector<std::string> filenames_;
  std::vector<int> labels_;
  bool has_labels_ = false;

  Index current_index_ = 0;

  int sequence_len_;
  int stride_;
  int step_;

  std::vector<VideoSampleDesc> samples_;
  CUDAStreamLease cuda_stream_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_BASE_H_
