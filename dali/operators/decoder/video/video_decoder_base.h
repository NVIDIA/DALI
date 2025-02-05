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
#include <optional>
#include "dali/kernels/common/copy.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

const int64_t kDefaultStartFrame = 0;
const int64_t kDefaultStride = 1;
const int64_t kDefaultSequenceLength = -1;

template <typename Backend, typename FramesDecoder>
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
      sample_shape[0] = sequence_length_[s] == -1 ? frames_decoders_[s]->NumFrames() :
                                                      sequence_length_[s];
      sample_shape[1] = frames_decoders_[s]->Height();
      sample_shape[2] = frames_decoders_[s]->Width();
      sample_shape[3] = frames_decoders_[s]->Channels();
      shape.set_tensor_shape(s, sample_shape);
    }
    return shape;
  }


  /**
   * Decode given number of frames from a FramesDecoder.
   *
   * This function allows optional padding of the output data, when
   * there's not enough frames inside FramesDecoder to fill the `num_frames` specified.
   *
   * The `pad_value` argument determines, with what the VideoDecoder shall pad
   * the partial sequence. The tensor passed to the `pad_value` shall be an entire
   * frame. This frame will be repeated at the end of partial sequence.
   * If the `pad_value` is not provided, the padding will not happen and the DecodeFrames
   * will return a partial sequence.
   *
   * @param output The SampleView in which the decoded sequence will be put.
   * @param sample_idx Index of the encoded video that shall be decoded.
   * @param num_frames How many frames shall be decoded.
   * @param pad_value What to (optionally) pad the partial sequence with.
   * @return False, if less than `num_frames` have been decoded.
   */
  bool DecodeFrames(SampleView<OutBackend> output, int64_t sample_idx,
                    int64_t start_frame, int64_t num_frames, int64_t stride,
                    std::optional<uint8_t> pad_value = std::nullopt,
                    std::optional<cudaStream_t> stream = std::nullopt) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();

    uint8_t *output_data = output.template mutable_data<uint8_t>();

    int64_t f = 0;
    if (start_frame > 0) {
      frames_decoder.SeekFrame(start_frame);
    }
    // Work until:
    //    (a) There are no more frames, or
    //    (b) Sufficient number of frames has been decoded.
    for (; f < num_frames && frames_decoder.NextFrameIdx() != -1; f++) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
      for (int i = 1; i < stride && frames_decoder.NextFrameIdx() != -1; i++) {
        frames_decoder.ReadNextFrame(nullptr, false);
      }
    }
    assert(f <= num_frames);
    bool full_sequence_decoded = f == num_frames;
    // If there's an insufficient number of frames, pad if requested.
    if (f < num_frames) {
      uint8_t use_pad_value = pad_value ? *pad_value : 0;
      auto pad_size = frame_size * (num_frames - f);
      if (std::is_same_v<OutBackend, CPUBackend>) {
        std::fill(output_data + f * frame_size, output_data + pad_size, use_pad_value);
      } else {
        cudaMemsetAsync(output_data + f * frame_size, use_pad_value, pad_size,
                        stream ? *stream : 0);
      }
    }
    return full_sequence_decoded;
  }


  /**
   * @brief Decode sample with index `idx` to `output` tensor.
   */
  void DecodeSample(SampleView<OutBackend> output, int64_t idx) {
    int64_t num_frames = sequence_length_[idx] == -1 ? output.shape()[0] : sequence_length_[idx];
    DecodeFrames(output, idx, start_frame_[idx], num_frames, stride_[idx]);
  }


  std::vector<std::unique_ptr<FramesDecoder>> frames_decoders_;
  std::vector<int64_t> start_frame_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> sequence_length_;
  bool if_build_index_;

 private:
  using storage_backend_for_copy_kernel = std::conditional_t<
          std::is_same_v<OutBackend, CPUBackend>,
          StorageCPU,
          StorageGPU
  >;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
