// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/reader/video_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(VideoReader, VideoReader, GPU);

DALI_SCHEMA(VideoReader)
  .DocStr(R"code(
Load and decode H264 video codec with FFmpeg and NVDECODE, NVIDIA GPU's hardware-accelerated video decoding.
The video codecs can be contained in most of container file formats. FFmpeg is used to parse video containers.
Returns a batch of sequences of `sequence_length` frames of shape [N, F, H, W, C] (N being the batch size and F the
number of frames).)code")
  .NumInput(0)
  .NumOutput(1)
  .AddArg("filenames",
      R"code(File names of the video files to load.)code",
      DALI_STRING_VEC)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).)code",
      -1)
  .AddOptionalArg("scale",
      R"code(Rescaling factor of height and width.)code",
      1.f)
  .AddOptionalArg("channels",
      R"code(Number of channels.)code",
      3)
  .AddOptionalArg("normalized",
      R"code(Get output as normalized data.)code",
      false)
  .AddOptionalArg("image_type",
      R"code(The color space of the output frames (supports RGB and YCbCr).)code",
      DALI_RGB)
  .AddOptionalArg("dtype",
      R"code(The data type of the output frames (supports FLOAT and UINT8).)code",
      DALI_UINT8)
  .AddParent("LoaderBase");
}  // namespace dali
