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

#ifndef DALI_OPERATORS_INPUT_VIDEO_INPUT_H_
#define DALI_OPERATORS_INPUT_VIDEO_INPUT_H_

#include <deque>
#include <string>
#include <vector>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

namespace dali {

namespace detail {
/**
 * TODO
 * @param num_frames How much frames the video has.
 * @param frames_per_sequence How much frames per sequence user requested.
 * @param batch_size How much sequences comprise a single batch.
 * @return
 */
auto DetermineBatchOutline(int num_frames, int frames_per_sequence, int batch_size) {
  assert(frames_per_sequence > 0);
  assert(batch_size > 0);
  // Initially, the Operator will return full batches of full sequences.
  const int num_full_batches = num_frames / frames_per_sequence / batch_size;
  const int remaining_frames = num_frames - num_full_batches * frames_per_sequence * batch_size;
  assert(remaining_frames < frames_per_sequence * batch_size);

  // Then, in the last batch, first few sequences will be full.
  const int num_full_sequences = remaining_frames / frames_per_sequence;
  assert(num_full_sequences < batch_size);

  // And at the end of the last batch, the last sequence will be either partial or padded,
  // depending on user setting.
  const int frames_in_last_sequence = remaining_frames - num_full_sequences * frames_per_sequence;
  assert(frames_in_last_sequence < frames_per_sequence);
  assert(num_full_batches >= 0 && num_full_sequences >= 0 && frames_in_last_sequence >= 0);
  return std::make_tuple(num_full_batches, num_full_sequences, frames_in_last_sequence);
}
}  // namespace detail

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
          last_sequence_policy_(spec.GetArgument<std::string>("last_sequence_policy")) {
    DALI_ENFORCE(last_sequence_policy_ == "partial" || last_sequence_policy_ == "pad",
                 make_string("Provided `last_sequence_policy` is not supported: ",
                             last_sequence_policy_));
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


  bool SetupImplDerived(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return true;
  }


  void RunImpl(Workspace &ws) override;

 private:
  void Invalidate() {
    valid_ = false;
    curr_frame_ = 0;
  }


  void DetermineOutputDescs(int num_frames) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(num_frames, frames_per_sequence_, batch_size_);
// cout<<"DetermineOuputDescs :"<<num_frames<<" "<<num_full_batches<<" "
// <<num_full_sequences<<" "<<frames_in_last_sequence<<endl;
    // Initially, the Operator will return full batches of full sequences.
    for (int i = 0; i < num_full_batches; i++) {
      OutputDesc od;
      od.shape = uniform_list_shape(batch_size_, GetSequenceShape(frames_per_sequence_, 0));
      od.type = DALI_UINT8;
      output_descs_.push_back(od);
    }

    if (num_full_sequences == 0 && frames_in_last_sequence == 0) {
      return;
    }

    OutputDesc od;
    od.type = DALI_UINT8;
    // Then, in the last batch, first few sequences will be full.
    od.shape = uniform_list_shape(
            num_full_sequences + (frames_in_last_sequence == 0 ? 0 : 1),
            GetSequenceShape(frames_per_sequence_, 0));
//    cout<<"TEMP: "<<num_full_sequences<<" "<<frames_in_last_sequence<<" "<<od.shape<<endl;

    if (frames_in_last_sequence != 0) {
      // And at the end of the last batch, the last sequence will be either partial or padded,
      // depending on user setting.
      od.shape.set_tensor_shape(od.shape.num_samples() - 1, GetSequenceShape(
              last_sequence_policy_ == "pad" ? frames_per_sequence_ : frames_in_last_sequence, 0));
    }
    output_descs_.push_back(od);
//    cout<<"DETERMINE OUTPUT DESCS\n";
//    for (auto i : output_descs_) {
//      cout<<i.shape<<" "<<i.type<<endl;
//    }
  }


  void InitializePadValue(uint8_t value) {
    pad_frame_shape_ = GetFrameShape(0);
    pad_value_data_ = std::vector<uint8_t>(pad_frame_shape_.num_elements(), value);
  }


  SampleView<Backend> GetPadFrame() {
    return {pad_value_data_.data(), pad_frame_shape_};
  }


  TensorShape<4> GetSequenceShape(int frames_per_sequence, int input_sample_idx) {
    TensorShape<4> ret;
    ret[0] = frames_per_sequence;
    ret[1] = this->frames_decoders_[input_sample_idx]->Height();
    ret[2] = this->frames_decoders_[input_sample_idx]->Width();
    ret[3] = this->frames_decoders_[input_sample_idx]->Channels();
    return ret;
  }


  TensorShape<3> GetFrameShape(int input_sample_idx) {
    TensorShape<3> ret;
    ret[0] = this->frames_decoders_[input_sample_idx]->Height();
    ret[1] = this->frames_decoders_[input_sample_idx]->Width();
    ret[2] = this->frames_decoders_[input_sample_idx]->Channels();
    return ret;
  }


  const int frames_per_sequence_ = {};
//  int one_over_frames_per_sequence_={};
  const int device_id_ = {};
  const int batch_size_ = {};
//  int one_over_batch_size_={};
  const std::string last_sequence_policy_;

  int curr_frame_ = 0;

  /// Valid VideoInput is the one that has the encoded video loaded and will return decoded sequence
  bool valid_ = false;

  std::deque<OutputDesc> output_descs_;
  OutputDesc output_desc_ = {};

  /// Buffer with the data used for padding incomplete sequences.
  std::vector<uint8_t> pad_value_data_ = {};
  TensorShape<3> pad_frame_shape_ = {};
};

}  // namespace dali

#endif  // DALI_OPERATORS_INPUT_VIDEO_INPUT_H_
