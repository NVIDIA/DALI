// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_

#include <string>
#include <vector>

#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/video/video_loader_decoder_cpu.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

namespace dali {
class VideoSampleGpu {
 public:
  void Decode();

  FramesDecoderGpu *video_file_ = nullptr;
  VideoSampleDesc *span_ = nullptr;
  int sequence_len_ = 0;
  Tensor<GPUBackend> data_;
  int label_ = -1;
};


class VideoLoaderDecoderGpu : public Loader<GPUBackend, VideoSampleGpu> {
 public:
  explicit inline VideoLoaderDecoderGpu(const OpSpec &spec) :
    Loader<GPUBackend, VideoSampleGpu>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    sequence_len_(spec.GetArgument<int>("sequence_length")),
    stride_(spec.GetArgument<int>("stride")),
    step_(spec.GetArgument<int>("step")) {
    InitCudaStream();
    if (step_ <= 0) {
      step_ = stride_ * sequence_len_;
    }
    has_labels_ = spec.TryGetRepeatedArgument(labels_, "labels");
  }

  void ReadSample(VideoSampleGpu &sample) override;

  void PrepareEmpty(VideoSampleGpu &sample) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override;

 private:
  void Reset(bool wrap_to_shard) override;

  void InitCudaStream();

  std::vector<std::string> filenames_;
  std::vector<int> labels_;
  bool has_labels_ = false;
  std::vector<FramesDecoderGpu> video_files_;
  std::vector<VideoSampleDesc> sample_spans_;

  Index current_index_ = 0;

  int sequence_len_;
  int stride_;
  int step_;

  CUDAStreamLease cuda_stream_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_
