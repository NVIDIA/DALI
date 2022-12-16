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
#include <vector>

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
    auto sample = encoded_videos[0];
    auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
    size_t size = sample.shape().num_elements();
    frames_decoders_[0] = std::make_unique<FramesDecoder>(data, size, false);

    assert(output_descs_.empty());
    DetermineOutputDescs(static_cast<int>(frames_decoders_[0]->NumFrames()));

    // This has to be done for every video file, since we need to know the shape of the frames.
    if (last_sequence_policy_ == "pad") {
      InitializePadValue(0);
    }

    valid_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_descs_.front();
  output_descs_.pop_front();
  return true;
}


template<>
void VideoInput<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  bool full_sequence;
  for (int64_t s = 0; s < output.num_samples(); s++) {
    auto pad_value =
            last_sequence_policy_ == "pad" ? std::optional<SampleView<CPUBackend>>(GetPadFrame())
                                           : std::nullopt;
    full_sequence = DecodeFrames(output[s], 0, frames_per_sequence_, pad_value);
    if (!full_sequence) {
      break;
    }
  }
  if (!full_sequence || frames_decoders_[0]->NextFrameIdx() == -1) {
    Invalidate();
  }
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
