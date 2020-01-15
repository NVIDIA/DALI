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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/video_reader_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(VideoReader, VideoReader, GPU);

DALI_SCHEMA(VideoReader)
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
  .AddOptionalArg("filenames",
      R"code(File names of the video files to load.
This option is mutually exclusive with `file_root` and `file_list`.)code",
      std::vector<std::string>{})
  .AddOptionalArg("file_root",
      R"code(Path to a directory containing data files.
This option is mutually exclusive with `filenames` and `file_list`.)code",
      std::string())
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of pairs ``file label``.
This option is mutually exclusive with `filenames` and `file_root`.)code",
      std::string())
  .AddOptionalArg("enable_frame_num",
      R"code(Return frame number output if file_list or file_root argument is passed)code",
      false)
  .AddOptionalArg("enable_timestamps",
      R"code(Return timestamps output if file_list or file_root argument is passed)code",
      false)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).)code",
      -1)
  .AddOptionalArg("channels",
      R"code(Number of channels.)code",
      3)
  .AddOptionalArg("additional_decode_surfaces",
      R"code(Additional decode surfaces to use beyond minimum required.
This is ignored when decoder is not able to determine minimum
number of decode surfaces, which may happen when using an older driver.
This parameter can be used trade off memory usage with performance.)code",
      2)
  .AddOptionalArg("normalized",
      R"code(Get output as normalized data.)code",
      false)
  .AddOptionalArg("image_type",
      R"code(The color space of the output frames (supports RGB and YCbCr).)code",
      DALI_RGB)
  .AddOptionalArg("dtype",
      R"code(The data type of the output frames (supports FLOAT and UINT8).)code",
      DALI_UINT8)
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in sequence.)code", 1u, false)
  .AddOptionalArg("skip_vfr_check",
      R"code(Skips check for variable frame rate on videos. This is useful when heuristic fails.)code", false)
  .AddOptionalArg("file_list_frame_num",
      R"code(If start/end timestamps are provided in file_list, interpret them as frame
numbers instead of timestamp. If floating point values are given, then
start frame number is ceiling of the number and end frame number is floor of
the number. Frame numbers start from 0.)code", false)
  .AddParent("LoaderBase");
}  // namespace dali
