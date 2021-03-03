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

DALI_REGISTER_OPERATOR(readers__VideoResize, VideoReaderResize, GPU);

DALI_SCHEMA(readers__VideoResize)
  .DocStr(R"code(Loads, decodes and resizes video files with FFmpeg and NVDECODE, which is
NVIDIA GPU's hardware-accelerated video decoding.

The video streams can be in most of the container file formats. FFmpeg is used to parse video
containers and returns a batch of sequences with shape ``(N, F, H, W, C)``, with N being
the batch size, and F the number of frames in the sequence.

This operator combines the features of :meth:`nvidia.dali.fn.video_reader` and :meth:`nvidia.dali.fn.resize`.

.. note::
  The decoder supports only constant frame-rate videos.
)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderOutputFn)
  .AddParent("VideoReader")
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");


// Deprecated alias
DALI_REGISTER_OPERATOR(VideoReaderResize, VideoReaderResize, GPU);

DALI_SCHEMA(VideoReaderResize)
    .DocStr("Legacy alias for :meth:`readers.video_resize`.")
    .NumInput(0)
    .OutputFn(detail::VideoReaderOutputFn)
    .AddParent("readers__VideoResize")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__VideoResize",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali
