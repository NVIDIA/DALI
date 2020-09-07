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
  .DocStr(R"code(Loads and decodes the H264, VP9, MPEG4 and HVEC(h265) video codec with FFmpeg
and NVDECODE, which is the hardware-accelerated video decoding feature in the NVIDIA(R)
graphical process unit (GPU).

The video codecs can be in most of the container file formats. FFmpeg is used to parse video
containers and returns a batch of sequences of sequence_length frames of the [N, F, H, W, C]
shape, where N is the batch size, and F is the number of frames). This class only supports
constant frame rate videos.)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderOutputFn)
  .AddOptionalArg("filenames",
      R"code(Path to the file with a list of the file label pairs.

This option is mutually exclusive with ``filenames`` and ``file_root``.)code",
      std::vector<std::string>{})
  .AddOptionalArg("file_root",
      R"code(Path to a directory that contains the data files.

This option is mutually exclusive with ``filenames`` and ``file_list``.)code",
      std::string())
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of ``file label [start_frame [end_frame]]``.

Positive value means the exact frame, negative counts as a Nth frame from the end (it follows
python array indexing schema), equal values for the start and end frame would yield an empty
sequence and a warning. This option is mutually exclusive with ``filenames``
and ``file_root``.)code",
      std::string())
  .AddOptionalArg("enable_frame_num",
      R"code(If the ``file_list`` or ``file_root`` argument is passed, returns
the frame number output.)code",
      false)
  .AddOptionalArg("enable_timestamps",
      R"code(If the`` file_list`` or ``file_root`` arguments are passed, returns
the timestamps output. )code",
      false)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence.

When the value is less than 0, ``step`` is set to ``sequence_length``.)code",
      -1)
  .AddOptionalArg("channels",
      R"code(Number of channels.)code",
      3)
  .AddOptionalArg("additional_decode_surfaces",
      R"code(Additional decode surfaces to use beyond minimum required.

This argument is ignored when the decoder cannot determine the minimum number of
decode surfaces

.. note::

  This can happen when the driver is an older version.

This parameter can be used to trade off memory usage with performance.)code",
      2)
  .AddOptionalArg("normalized",
      R"code(Gets the output as normalized data.)code",
      false)
  .AddOptionalArg("image_type",
      R"code(The color space of the output frames, and RGB and YCbCr are supported.)code",
      DALI_RGB)
  .AddOptionalArg("dtype",
      R"code(The data type of the output frames, and  FLOAT and UINT8 are supported.)code",
      DALI_UINT8)
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in the sequence.)code", 1u, false)
  .AddOptionalArg("skip_vfr_check",
      R"code(Skips the check for the variable frame rate on videos.

This argument is useful when heuristic fails.)code", false)
  .AddOptionalArg("file_list_frame_num",
      R"code(If the start/end timestamps are provided in file_list, you can interpret them
as frame numbers instead of as timestamps.

If floating point values have been provided, the start frame number is the highest possible number,
and the end frame number is the lowest possible number.

Frame numbers start from 0.)code", false)
  .AddParent("LoaderBase");
}  // namespace dali
