// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/video/video_loader_decoder_cpu.h"

namespace dali {

void VideoLoaderDecoderCpu::PrepareEmpty(VideoSample<CPUBackend> &sample) {
  sample = {};
  sample.data_.set_pinned(false);
  sample.data_.SetLayout("FHWC");
}

void VideoLoaderDecoderCpu::ReadSample(VideoSample<CPUBackend> &sample) {
  auto &sample_span = sample_spans_[current_index_];
  auto &video_file = video_files_[sample_span.video_idx_];

  ++current_index_;
  MoveToNextShard(current_index_);

  sample.data_.Resize(
    TensorShape<4>{sequence_len_, video_file.Height(), video_file.Width(), video_file.Channels()},
    DALIDataType::DALI_UINT8);
  sample.data_.SetSourceInfo(video_file.Filename());

  auto data = sample.data_.mutable_data<uint8_t>();

  // TODO(awolant): Extract decoding outside of ReadSample (ReaderDecoder abstraction)
  for (int i = 0; i < sequence_len_; ++i) {
    video_file.SeekFrame(sample_span.start_ + i * sample_span.stride_);
    video_file.ReadNextFrame(data + i * video_file.FrameSize());
  }

  if (has_labels_) {
    sample.label_ = labels_[sample_span.video_idx_];
  }
  if (has_frame_idx_) {
    sample.first_frame_ = sample_span.start_;
  }
}

Index VideoLoaderDecoderCpu::SizeImpl() {
  return sample_spans_.size();
}

void VideoLoaderDecoderCpu::Skip() {
  MoveToNextShard(++current_index_);
}

void VideoLoaderDecoderCpu::PrepareMetadataImpl() {
  video_files_.reserve(filenames_.size());
  for (auto &filename : filenames_) {
    video_files_.emplace_back(filename);
    if (!video_files_.back().IsValid()) {
      LOG_LINE << "Invalid video file: " << filename << std::endl;
      video_files_.pop_back();
    } else {
      LOG_LINE << "Valid video file: " << filename << std::endl;
    }
  }

  for (size_t video_idx = 0; video_idx < video_files_.size(); ++video_idx) {
    for (int start = 0;
         start + stride_ * sequence_len_ <= video_files_[video_idx].NumFrames();
         start += step_) {
      sample_spans_.push_back(
        VideoSampleDesc(start, start + stride_ * sequence_len_, stride_, video_idx));
    }
  }
  if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(std::begin(sample_spans_), std::end(sample_spans_), g);
    }

    // set the initial index for each shard
    Reset(true);
}

void VideoLoaderDecoderCpu::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(virtual_shard_id_, num_shards_, SizeImpl()) : 0;
}

}  // namespace dali
