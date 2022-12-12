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

#include "dali/operators/input/video_input.h"
#include <memory>

namespace dali {

/// By definition, the input batch size of this Operator is always 1.
constexpr int input_batch_size = 1;


template<>
bool VideoInput<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  if (!valid_) {
    InputOperator<CPUBackend>::HandleDataAvailability();
    TensorList<CPUBackend> encoded_videos;
    frames_decoders_.resize(input_batch_size);
    auto &thread_pool = ws.GetThreadPool();
    this->ForwardCurrentData(encoded_videos, thread_pool);

    // Creating FramesDecoders
    for (int i = 0; i < input_batch_size; ++i) {
      auto sample = encoded_videos[i];
      auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
      size_t size = sample.shape().num_elements();
      frames_decoders_[i] = std::make_unique<FramesDecoder>(data, size, false);
    }
    auto sequence_shape = GetSequenceShape(frames_per_sequence_, 0);
    output_desc_.shape = uniform_list_shape(batch_size_, sequence_shape);
    output_desc_.type = DALI_UINT8;

    if (last_sequence_policy_ == "pad") {
      InitializePadValue(0, sequence_shape);
    }

    valid_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_desc_;
  cout << "SETUP: Output shape " << output_desc[0].shape << " TYPE " << output_desc[0].type
       << endl;
  return true;
}


template<>
void VideoInput<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  bool full_sequence;
  for (int64_t s = 0; s < batch_size_; s++) {
    cout << "Curr frame " << curr_frame_ << " fps " << frames_per_sequence_ << endl;
    auto pad_value = last_sequence_policy_ == "pad" ? std::optional<SampleView<CPUBackend>>(
            GetPadFrame(output_desc_.shape[s])) : std::nullopt;
    full_sequence = DecodeFrames(output[s], 0, frames_per_sequence_, pad_value);
    if (!full_sequence) break;
    curr_frame_ += frames_per_sequence_;
  }
  if (!full_sequence || frames_decoders_[0]->NextFrameIdx() == -1) {
    Invalidate();
  }
  if (last_sequence_policy_ == "pad") {
    PadSequence();
  }
  cout << "Output shape: " << output.shape() << endl;
//  if (!CanDecode(0)) {
//    auto h = horcruxes_.front();
//    horcruxes_.pop();
//    return_horcruxes_.emplace_back(h);
//  }
}


template<>
bool VideoInput<GPUBackend>::SetupImplDerived(std::vector<OutputDesc> &output_desc,
                                              const Workspace &ws) {
  DALI_FAIL("Shouldn't be called");
  return true;
}


template<>
void VideoInput<GPUBackend>::RunImpl(Workspace &ws) {
  DALI_FAIL("Shouldn't be called");
}


DALI_SCHEMA(experimental__inputs__Video)
                .DocStr(
                        R"code(...)code")
                .NumInput(0)
                .NumOutput(1)
                .AddArg("frames_per_sequence", R"code(...)code", DALI_INT32)
                .AddOptionalArg("last_sequence_policy", R"code(...)code", "partial")
                .AddParent("InputOperatorBase");


DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<CPUBackend>, CPU);

}  // namespace dali