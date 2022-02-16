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

#ifndef DALI_OPERATORS_READER_VIDEO_READER_DECODER_GPU_OP_H_
#define DALI_OPERATORS_READER_VIDEO_READER_DECODER_GPU_OP_H_

#include <vector>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/video/video_loader_decoder_gpu.h"

namespace dali {
class VideoReaderDecoderGpu : public DataReader<GPUBackend, VideoSampleGpu> {
 public:
  explicit VideoReaderDecoderGpu(const OpSpec &spec);

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) override;

  void RunImpl(DeviceWorkspace &ws) override;

  bool CanInferOutputs() const override { return true; }

  void Prefetch() override;

 private:
  bool has_labels_ = false;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_DECODER_GPU_OP_H_
