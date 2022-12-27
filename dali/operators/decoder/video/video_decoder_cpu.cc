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

#include "dali/operators/decoder/video/video_decoder_cpu.h"

namespace dali {

bool VideoDecoderCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const Workspace &ws) {
  ValidateInput(ws);
  const auto &input = ws.Input<CPUBackend>(0);
  int batch_size = input.num_samples();
  frames_decoders_.resize(batch_size);
  auto &thread_pool = ws.GetThreadPool();
  for (int i = 0; i < batch_size; ++i) {
    auto sample = input[i];
    auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
    size_t size = sample.shape().num_elements();
    thread_pool.AddWork([this, i, data, size](int tid) {
      frames_decoders_[i] = std::make_unique<FramesDecoder>(data, size, false);
    });
  }
  thread_pool.RunAll();
  output_desc.resize(1);
  output_desc[0].shape = ReadOutputShape();
  output_desc[0].type = DALI_UINT8;
  return true;
}

void VideoDecoderCpu::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  const auto &input = ws.Input<CPUBackend>(0);
  int batch_size = input.num_samples();
  auto &thread_pool = ws.GetThreadPool();
  for (int s = 0; s < batch_size; ++s) {
    thread_pool.AddWork([this, s, &output](int tid) {
      DecodeSample(output[s], s);
    }, volume(input[s].shape()));
  }
  thread_pool.RunAll();
}

DALI_SCHEMA(experimental__decoders__Video)
    .DocStr(
        R"code(Decodes a video file from a memory buffer (e.g. provided by external source).

The video streams can be in most of the container file formats. FFmpeg is used to parse video
 containers and returns a batch of sequences of frames with shape (F, H, W, C) where F is the
 number of frames in a sequence and can differ for each sample.)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "buffer", "TensorList", "Data buffer with a loaded video file.")
    .AddOptionalArg("affine",
    R"code(Applies only to the mixed backend type.

If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
 Otherwise, the threads can be reassigned to any CPU core by the operating system.)code", true);

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderCpu, CPU);

}  // namespace dali
