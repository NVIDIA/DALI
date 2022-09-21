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

#include <vector>
#include <memory>

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_H_

namespace dali {

template <typename Backend>
struct FramesDecoderImpl {};

template <>
struct FramesDecoderImpl<CPUBackend> {
  using type = FramesDecoder;
};

template <>
struct FramesDecoderImpl<GPUBackend> {
  using type = FramesDecoderGpu;
};

template <typename Backend>
using FramesDecoder_t = typename FramesDecoderImpl<Backend>::type;

template <typename Backend>
class DLL_PUBLIC VideoDecoder : public Operator<Backend> {
 public:
  explicit VideoDecoder(const OpSpec &spec)
      : Operator<Backend>(spec) {}

  ~VideoDecoder() override = default;
  DISABLE_COPY_MOVE_ASSIGN(VideoDecoder);

  USE_OPERATOR_MEMBERS();

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:
  void ValidateInput(const workspace_t<Backend> &ws) {
    const auto &input = ws.template Input<Backend>(0);
    DALI_ENFORCE(input.type() == DALI_UINT8,
                 "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1,
                 "Input buffer must be 1-dimensional.");
  }

  TensorListShape<4> ReadOutputShape() {
    TensorListShape<4> shape(frames_decoders_.size());
    for (size_t s = 0; s < frames_decoders_.size(); ++s) {
      TensorShape<4> sample_shape;
      sample_shape[0] = frames_decoders_[s]->NumFrames();
      sample_shape[1] = frames_decoders_[s]->Height();
      sample_shape[2] = frames_decoders_[s]->Width();
      sample_shape[3] = frames_decoders_[s]->Channels();
      shape.set_tensor_shape(s, sample_shape);
    }
    return shape;
  }

  void DecodeSample(SampleView<Backend> output, FramesDecoder_t<Backend> &frames_decoder) {
    int64_t num_frames = output.shape()[0];
    int64_t frame_size = frames_decoder.FrameSize();
    uint8_t *output_data = output.template mutable_data<uint8_t>();
    for (int f = 0; f < num_frames; ++f) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
    }
  }

  std::vector<std::unique_ptr<FramesDecoder_t<Backend>>> frames_decoders_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_H_
