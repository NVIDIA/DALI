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
  .AddOptionalArg("cache_size",
      R"code(Total size of the decoder cache in megabytes. When provided, decoded
images bigger than `cache_threshold` will be cached in memory.)code",
      0)
  .AddOptionalArg("cache_threshold",
      R"code(Size threshold (in bytes) for images to be cached.)code",
      0)
  .AddOptionalArg("cache_debug",
      R"code(Print debug information about decoder cache.)code",
      false)
  .AddOptionalArg("cache_type",
      R"code(Choose cache type:
`threshold`: Caches every image with size bigger than `cache_threshold` until cache is full.
Warm up time for `threshold` policy is 1 epoch.
`largest`: Store largest images that can fit the cache.
Warm up time for `largest` policy is 2 epochs
default: `largest`.
To take advantage of caching, it is recommended to use the option `stick_to_shard=True` with
the reader operators, to limit the amount of unique images seen by the decoder in a multi node environment)code",
      std::string());

}  // namespace dali
