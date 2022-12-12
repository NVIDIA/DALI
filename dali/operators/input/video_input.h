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

#ifndef DALI_VIDEO_INPUT_H
#define DALI_VIDEO_INPUT_H

#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

namespace dali {

template<typename Backend>
using frames_decoder_t =
        std::conditional_t<
                std::is_same<Backend, CPUBackend>::value,
                FramesDecoder,
                FramesDecoderGpu
        >;

template<typename Backend, typename FramesDecoder = frames_decoder_t<Backend>>
class VideoInput : public VideoDecoderBase<Backend, FramesDecoder>, public InputOperator<Backend> {
//  using VideoDecoderBase<Backend, FramesDecoder>::frames_decoders_;
 public:
  explicit VideoInput(const OpSpec &spec) :
          InputOperator<Backend>(spec),
          frames_per_sequence_(spec.GetArgument<int>("frames_per_sequence")),
          device_id_(spec.GetArgument<int>("device_id")),
          batch_size_(spec.GetArgument<int>("max_batch_size")),
          last_sequence_policy_(spec.GetArgument<std::string>("last_sequence_policy")){
    Invalidate();
  }


  bool CanInferOutputs() const override {
    return true;
  }


  int NextBatchSize() override {
    return batch_size_;
  }


  void Advance() override {
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;


  bool SetupImplDerived(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {}


  void RunImpl(Workspace &ws) override;

 private:
  void Invalidate() {
    valid_ = false;
    curr_frame_ = 0;
  }

  void InitializePadValue(uint8_t value, const TensorShape<3>& frame_shape) {
    auto num_values = frame_shape.num_elements();
    pad_value_data_=std::vector<uint8_t>(num_values,value);
  }

  SampleView<Backend> GetPadFrame(const TensorShape<3>& frame_shape) {
    return {pad_value_data_.data(), frame_shape};
  }

  void PadSequence() {

  }

  TensorShape<4> GetSequenceShape(int frames_per_sequence, int input_sample_idx) {
    TensorShape<4> sample_shape;
    sample_shape[0] = frames_per_sequence;
    sample_shape[1] = this->frames_decoders_[input_sample_idx]->Height();
    sample_shape[2] = this->frames_decoders_[input_sample_idx]->Width();
    sample_shape[3] = this->frames_decoders_[input_sample_idx]->Channels();
    return sample_shape;
  }



  int frames_per_sequence_={};
  int device_id_={};
  int batch_size_={};
  std::string last_sequence_policy_={};

  int curr_frame_=0;

  /// Valid VideoInput is the one that has the encoded video loaded and will return decoded sequence
  bool valid_ = false;

  OutputDesc output_desc_ = {};

  /// Buffer with the data used for padding incomplete sequences.
  std::vector<uint8_t> pad_value_data_={};
};

}

#endif //DALI_VIDEO_INPUT_H
