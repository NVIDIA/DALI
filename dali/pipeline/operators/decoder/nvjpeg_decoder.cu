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

#include <string>
#include "dali/pipeline/operators/decoder/nvjpeg_decoder.h"

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);

DALI_SCHEMA(nvJPEGDecoder)
  .DocStr(R"code(Decode JPEG images using the nvJPEG library.
Output of the decoder is on the GPU and uses `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(The color space of output image.)code",
      DALI_RGB)
  .AddOptionalArg("use_batched_decode",
      R"code(Use nvJPEG's batched decoding API.)code",
      false)
  .AddOptionalArg("device_memory_padding",
      R"code(Padding for nvJPEG's device memory allocations in bytes.
This parameter helps to avoid reallocation in nvJPEG whenever a bigger image
is encountered and internal buffer needs to be reallocated to decode it.)code",
      16*1024*1024)
  .AddOptionalArg("host_memory_padding",
      R"code(Padding for nvJPEG's host memory allocations in bytes.
This parameter helps to avoid reallocation in nvJPEG whenever a bigger image
is encountered and internal buffer needs to be reallocated to decode it.)code",
      16*1024*1024)
  .AddParent("CachedDecoderAttr");

}  // namespace dali
