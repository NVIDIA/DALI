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

#include "dali/operators/decoder/video/video_decoder_mixed.h"

namespace dali {

bool VideoDecoderMixed::SetupImpl(
  std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  ValidateInput(ws);
  const auto &input = ws.Input<CPUBackend>(0);
  int batch_size = input.num_samples();
  auto stream = ws.stream();
  frames_decoders_.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    thread_pool_.AddWork([this, i, &input, stream](int) {
      auto sample = input[i];
      auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
      size_t size = sample.shape().num_elements();
      auto source_info = input.GetMeta(i).GetSourceInfo();
      frames_decoders_[i] = std::make_unique<FramesDecoderGpu>(data, size, stream, false, -1,
                                                               source_info);
    });
  }
  thread_pool_.RunAll();
  for (auto &dec : frames_decoders_) {
    DALI_ENFORCE(dec->IsValid(), make_string("Failed to create video decoder for \"",
                                 dec->Filename(), "\""));
  }
  output_desc.resize(1);
  output_desc[0].shape = ReadOutputShape();
  output_desc[0].type = DALI_UINT8;
  return true;
}

void VideoDecoderMixed::RunImpl(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);
  const auto &input = ws.Input<CPUBackend>(0);
  int batch_size = input.num_samples();
  for (int s = 0; s < batch_size; ++s) {
    thread_pool_.AddWork([this, s, &output](int) {
      DecodeSample(output[s], s);
      // when the decoding is done release the decoder,
      // so it can be reused by the next sample in the batch
      frames_decoders_[s].reset();
    }, input[s].shape().num_elements());
  }
  thread_pool_.RunAll();
}

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderMixed, Mixed);

}  // namespace dali
