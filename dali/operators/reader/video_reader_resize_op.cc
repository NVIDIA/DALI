// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/video_reader_op.h"
#include "dali/operators/reader/video_reader_resize_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(VideoReaderResize, VideoReaderResize, GPU);

DALI_SCHEMA(VideoReaderResize)
  .DocStr(R"code(
Load and decode H264 video codec with FFmpeg and NVDECODE, NVIDIA GPU's hardware-accelerated video decoding.
The video codecs can be contained in most of container file formats. FFmpeg is used to parse video containers.
Returns a batch of sequences of `sequence_length` frames of shape [N, F, H, W, C] (N being the batch size and F the
number of frames). Supports only constant frame rate videos.)code")
  .NumInput(0)
  .OutputFn([](const OpSpec &spec) {
      std::string file_root = spec.GetArgument<std::string>("file_root");
      std::string file_list = spec.GetArgument<std::string>("file_list");
      bool enable_frame_num = spec.GetArgument<bool>("enable_frame_num");
      bool enable_timestamps = spec.GetArgument<bool>("enable_timestamps");
      int num_outputs = 1;
      if (!file_root.empty() || !file_list.empty()) {
        num_outputs++;
        if (enable_frame_num) num_outputs++;
        if (enable_timestamps) num_outputs++;
      }
      return num_outputs;
    })
  .AddOptionalArg("resize",
       R"code(Resize video while loading)code", false)
  .AddOptionalArg("resize_x", R"code(The length of the X dimension of the resized image.
If the `resize_y` is left at 0, then the op will keep
the aspect ratio of the original image.)code", 0.f, true)
  .AddOptionalArg("resize_y", R"code(The length of the Y dimension of the resized image.
If the `resize_x` is left at 0, then the op will keep
the aspect ratio of the original image.)code", 0.f, true)
  .AddOptionalArg("interp_type",
       R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR)
  .AddParent("VideoReader");
}  // namespace dali
