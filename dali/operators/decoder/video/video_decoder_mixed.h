// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_MIXED_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_MIXED_H_

#include <vector>
#include <memory>
#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

class VideoDecoderMixed
        : public Operator<MixedBackend>, public VideoDecoderBase<MixedBackend, FramesDecoderGpu> {
  using Operator<MixedBackend>::num_threads_;
  using VideoDecoderBase::DecodeSample;

 public:
  explicit VideoDecoderMixed(const OpSpec &spec):
    Operator<MixedBackend>(spec),
    thread_pool_(num_threads_,
                 spec.GetArgument<int>("device_id"),
                 spec.GetArgument<bool>("affine"),
                 "mixed video decoder") {}



  void RunImpl(Workspace &ws) override;

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override;

 private:
  void AcquireArguments(const Workspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    if (this->spec_.ArgumentDefined("start_frame")) {
      this->GetPerSampleArgument(start_frame_, "start_frame", ws, curr_batch_size);
      if_build_index_ = true;
    } else {
      start_frame_ = std::vector<int64_t>(curr_batch_size, kDefaultStartFrame);
    }
    if (this->spec_.ArgumentDefined("stride")) {
      this->GetPerSampleArgument(stride_, "stride", ws, curr_batch_size);
      if_build_index_ = true;
    } else {
      stride_ = std::vector<int64_t>(curr_batch_size, kDefaultStride);
    }
    if (this->spec_.ArgumentDefined("sequence_length")) {
      this->GetPerSampleArgument(sequence_length_, "sequence_length", ws, curr_batch_size);
      if_build_index_ = true;
    } else {
      sequence_length_ = std::vector<int64_t>(curr_batch_size, kDefaultSequenceLength);
    }
  }

  ThreadPool thread_pool_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_MIXED_H_
