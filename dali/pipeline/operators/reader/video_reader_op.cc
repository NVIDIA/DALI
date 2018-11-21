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
The video codecs can be contained in most of container file formats.
Returns a batch of sequences of `count` frames of shape [N, S, H, W, C] (N being the batch size and S the
number of frames).)code")
  .NumInput(0)
  .NumOutput(1)
  .AddArg("filenames",
      R"code(File names of the video files to load.)code",
      DALI_STRING_VEC)
  .AddArg("count",
      R"code(Frames to load per batch.)code",
      DALI_INT32)
  .AddArg("height",
      R"code(Height of the desired frames.)code",
      DALI_INT32)
  .AddArg("width",
      R"code(Width of the desired frames.)code",
      DALI_INT32)
  .AddOptionalArg("channels",
      R"code(Number of channels.)code",
      3)
  .AddParent("LoaderBase");
}  // namespace dali
