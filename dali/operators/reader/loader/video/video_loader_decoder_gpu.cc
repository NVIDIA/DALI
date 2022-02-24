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

#include "dali/operators/reader/loader/video/video_loader_decoder_gpu.h"

#include "dali/util/nvml.h"

namespace dali {
void VideoSampleGpu::Decode() {
  TensorShape<4> shape = {
    sequence_len_,
    video_file_->Height(),
    video_file_->Width(),
    video_file_->Channels()};

  data_.Resize(
    shape,
    DALIDataType::DALI_UINT8);

  for (int i = 0; i < sequence_len_; ++i) {
    int frame_id = span_->start_ + i * span_->stride_;
    video_file_->SeekFrame(frame_id);
    video_file_->ReadNextFrame(
      static_cast<uint8_t *>(data_.raw_mutable_data()) + i * video_file_->FrameSize());
  }
}

void VideoLoaderDecoderGpu::InitCudaStream() {
  #if NVML_ENABLED
  {
    nvml::Init();
    static float driver_version = nvml::GetDriverVersion();
    if (driver_version > 460 && driver_version < 470.21) {
      DALI_WARN_ONCE("Warning: Decoding on a default stream. Performance may be affected.");
      return;
    }
  }
  #else
  {
    int driver_cuda_version = 0;
    CUDA_CALL(cuDriverGetVersion(&driver_cuda_version));
    if (driver_cuda_version >= 11030 && driver_cuda_version < 11040) {
      DALI_WARN_ONCE("Warning: Decoding on a default stream. Performance may be affected.");
      return;
    }
  }
  #endif

  // TODO(awolant): Check per decoder stream
  cuda_stream_ = CUDAStreamPool::instance().Get(device_id_);
}

void VideoLoaderDecoderGpu::PrepareEmpty(VideoSampleGpu &sample) {
  sample = {};
}

void VideoLoaderDecoderGpu::ReadSample(VideoSampleGpu &sample) {
  auto &sample_span = sample_spans_[current_index_];

  // Bind sample to the video and span, so it can be decoded later
  sample.span_ = &sample_span;
  sample.video_file_ = &video_files_[sample_span.video_idx_];
  sample.sequence_len_ = sequence_len_;

  if (has_labels_) {
    sample.label_ = labels_[sample_span.video_idx_];
  }

  ++current_index_;
  MoveToNextShard(current_index_);
}

Index VideoLoaderDecoderGpu::SizeImpl() {
  return sample_spans_.size();
}

void VideoLoaderDecoderGpu::PrepareMetadataImpl() {
  video_files_.reserve(filenames_.size());
  for (auto &filename : filenames_) {
    video_files_.emplace_back(filename, cuda_stream_);
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

void VideoLoaderDecoderGpu::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
}

}  // namespace dali
