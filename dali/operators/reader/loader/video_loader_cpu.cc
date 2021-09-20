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

#include "dali/operators/reader/loader/video_loader_cpu.h"

namespace dali {

void VideoLoaderCPU::ReadSample(Tensor<CPUBackend> &sample) {
  auto &sample_span = sample_spans_[current_index_];
  auto &video_file = video_files_[sample_span.video_idx_];

  ++current_index_;
  MoveToNextShard(current_index_);

  sample.set_type(DALI_UINT8);
  sample.Resize(
    TensorShape<4>{sequence_len_, video_file.Width(), video_file.Height(), video_file.Channels()});

  auto data = sample.mutable_data<uint8_t>();

  for (int i = 0; i < sequence_len_; ++i) {
    video_file.SeekFrame(sample_span.start_ + i * sample_span.stride_);     //This seek can be optimized - for consecutive frames not needed etc.
    video_file.ReadNextFrame(data + i * video_file.FrameSize());
  }
}

Index VideoLoaderCPU::SizeImpl() {
  return sample_spans_.size();
}

void VideoLoaderCPU::PrepareMetadataImpl() {
  for (auto &filename : filenames_) {
    video_files_.push_back(VideoFileCPU(filename));
  }

  for (int video_idx = 0; video_idx < video_files_.size(); ++video_idx) {
    for (int start = 0; start + stride_ * sequence_len_ <= video_files_[video_idx].NumFrames(); start += step_) {
      sample_spans_.push_back(VideoSampleDesc(start, start + stride_ * sequence_len_, stride_, video_idx));
    }
  }
}

void VideoLoaderCPU::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
}

}  // namespace dali
