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
number of frames). Supports only constant frame rate videos. It resizes video based on provided params. It supports
features of `Resize` operator.)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderOutputFn)
  .AddParent("VideoReader")
  .AddParent("ResizeAttr")
  .AddParent("ResamplingFilterAttr");
}  // namespace dali
