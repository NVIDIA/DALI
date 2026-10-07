// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_READER_VIDEO_READER_DECODER_RESIZE_OP_H_
#define DALI_OPERATORS_VIDEO_READER_VIDEO_READER_DECODER_RESIZE_OP_H_

#include <vector>

#include "dali/operators/image/resize/resampling_attr.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/operators/video/reader/video_reader_decoder_op.h"

namespace dali {

/**
 * @brief experimental.readers.video with a fused resize of the decoded frames.
 *
 * The decoding (Prefetch) and the metadata outputs are inherited from VideoReaderDecoder;
 * only the video output is different: instead of being copied, the decoded sequences are
 * resized directly from the prefetched samples into output 0.
 */
class VideoReaderDecoderResize : public VideoReaderDecoder<GPUBackend>,
                                 protected ResizeBase<GPUBackend> {
 public:
  explicit VideoReaderDecoderResize(const OpSpec &spec);

  ~VideoReaderDecoderResize() override = default;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

 private:
  void ResizeVideoOutput(Workspace &ws);

  ResizeAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;
  std::vector<kernels::ResamplingParams2D> resample_params_;
  TensorListShape<> input_shape_, output_shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_READER_VIDEO_READER_DECODER_RESIZE_OP_H_
