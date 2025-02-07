// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"

namespace dali {

DALI_SCHEMA(experimental__decoders__Video)
    .DocStr(
        R"code(Decodes video files from memory buffers into sequences of frames.

The operator accepts video files in common container formats (e.g. MP4, AVI). For CPU backend,
FFmpeg is used for decoding. For Mixed backend, NVIDIA's Video Codec SDK (NVDEC) is used.

Each output sample is a sequence of frames with shape (F, H, W, C) where:
- F is the number of frames in the sequence (can vary between samples)
- H is the frame height in pixels
- W is the frame width in pixels 
- C is the number of color channels)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "buffer", "TensorList", "Memory buffer containing the encoded video file data")
    .AddOptionalArg("affine",
    R"code(Whether to pin threads to CPU cores (mixed backend only).

If True, each thread in the internal thread pool will be pinned to a specific CPU core.
If False, threads can migrate between cores based on OS scheduling.)code", true)
    .AddOptionalArg<int>("start_frame",
    R"code(Index of the first frame to extract from each video)code", 0, true)
    .AddOptionalArg<int>("stride",
    R"code(Number of frames to skip between each extracted frame)code", 1, true)
    .AddOptionalArg<int>("sequence_length",
    R"code(Number of frames to extract from each video. If not provided, the whole video is decoded.)code", nullptr, true)
    .AddOptionalArg<std::string>("pad_mode",
    R"code(How to handle videos with insufficient frames:
- none: Return shorter sequences if not enough frames
- constant: Pad with a fixed value (specified by ``pad_value``)
- edge: Repeat the last valid frame)code", "constant", true)
    .AddOptionalArg<int>("pad_value",
    R"code(Value used to pad missing frames when pad_mode='constant'. Must be in range [0, 255].)code", 0);

class VideoDecoderCpu : public VideoDecoderBase<CPUBackend, FramesDecoder> {
 public:
  explicit VideoDecoderCpu(const OpSpec &spec) :
    VideoDecoderBase<CPUBackend, FramesDecoder>(spec) {}
};

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderCpu, CPU);

}  // namespace dali
