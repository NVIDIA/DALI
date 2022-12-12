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

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_

#include <vector>
#include <memory>
#include "dali/kernels/common/copy.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

template<typename Backend, typename FramesDecoder>
class DLL_PUBLIC VideoDecoderBase {
 public:
  using InBackend = CPUBackend;
  using OutBackend = std::conditional_t<std::is_same_v<Backend, CPUBackend>,
          CPUBackend,
          GPUBackend>;

  VideoDecoderBase() = default;

  DISABLE_COPY_MOVE_ASSIGN(VideoDecoderBase);

 protected:
  void ValidateInput(const Workspace &ws) {
    const auto &input = ws.Input<InBackend>(0);
    DALI_ENFORCE(input.type() == DALI_UINT8,
                 "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1,
                 "Input buffer must be 1-dimensional.");
    for (int64_t i = 0; i < input.num_samples(); ++i) {
      DALI_ENFORCE(input[i].shape().num_elements() > 0,
                   make_string("Incorrect sample at position: ", i, ". ",
                               "Video decoder does not support empty input samples."));
    }
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


  /**
   *
   * @param output The SampleView in which the decoded sequence will be put.
   * @param sample_idx Index of the encoded video that shall be decoded.
   * @param num_frames How many frames shall be decoded.
   * @param pad_value What to (optionally) pad the partial sequence with.
   * @return False, if less than `num_frames` have been decoded.
   */
  bool DecodeFrames(SampleView<OutBackend> output, int64_t sample_idx, int64_t num_frames,
                    std::optional<SampleView<OutBackend>> pad_value = std::nullopt,
                    cudaStream_t stream = 0) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();

    DALI_ENFORCE(!pad_value.has_value() || pad_value->shape().num_elements() == frame_size,
                 make_string("Provided pad_value has improper number of elements. Expected: ",
                             frame_size, "; Actual: ", pad_value->shape().num_elements()));

    uint8_t *output_data = output.template mutable_data<uint8_t>();
    //TODO seek frame

    int64_t f = 0;
    // Work until:
    //      1. There are no more frames, or
    //      2. Sufficient number of frames has been decoded.
    for (; f < num_frames && frames_decoder.NextFrameIdx() != -1; f++) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
    }
    bool full_sequence_decoded = f >= num_frames;
    // If there's an insufficient number of frames, pad if requested.
    for (; f < num_frames && pad_value.has_value(); f++) {
      kernels::copy<OutBackend, OutBackend>(
              output_data + f * frame_size, pad_value->raw_data(), frame_size, stream);
    }
    return full_sequence_decoded;
  }


  /**
   * @brief Decode sample with index `idx` to `output` tensor.
   */
  void DecodeSample(SampleView<OutBackend> output, int64_t idx) {
    int64_t num_frames = output.shape()[0];
    DecodeFrames(output, idx, num_frames);
  }


  std::vector<std::unique_ptr<FramesDecoder>> frames_decoders_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
