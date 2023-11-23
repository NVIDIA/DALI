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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_

#include <string>
#include <vector>

#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/video/video_loader_decoder_base.h"
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
  std::string &FileName();
};


class VideoLoaderDecoderGpu
  : public Loader<GPUBackend, VideoSampleGpu, true>, VideoLoaderDecoderBase {
 public:
  explicit inline VideoLoaderDecoderGpu(const OpSpec &spec) :
    Loader<GPUBackend, VideoSampleGpu, true>(spec),
    VideoLoaderDecoderBase(spec) {
    InitCudaStream();
  }

  void ReadSample(VideoSampleGpu &sample) override;

  void PrepareEmpty(VideoSampleGpu &sample) override;

  void Skip() override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override;

 private:
  void Reset(bool wrap_to_shard) override;

  void InitCudaStream();

  std::vector<FramesDecoderGpu> video_files_;

  CUDAStreamLease cuda_stream_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_LOADER_DECODER_GPU_H_
