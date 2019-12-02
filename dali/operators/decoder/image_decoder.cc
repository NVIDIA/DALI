// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/operator.h"
#include "dali/operators/decoder/host/host_decoder.h"

namespace dali {

// ImageDecoder common attributes (does not include Cached decoder attributes which are present in
// ImageDecoder but not on the fused variants)
DALI_SCHEMA(ImageDecoderAttr)
  .DocStr(R"code(Image decoder common attributes)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(The color space of output image.)code",
      DALI_RGB)
// TODO(janton): Remove this when we remove the old nvJPEGDecoder implementation (DALI-971)
#if !defined(NVJPEG_DECOUPLED_API)
  .AddOptionalArg("use_batched_decode",
      R"code(**`mixed` backend only** Use nvJPEG's batched decoding API.)code", false)
#endif
  .AddOptionalArg("hybrid_huffman_threshold",
      R"code(**`mixed` backend only** Images with number of pixels (height * width) above this threshold will use the nvJPEG hybrid Huffman decoder.
Images below will use the nvJPEG full host huffman decoder.
N.B.: Hybrid Huffman decoder still uses mostly the CPU.)code",
      1000u*1000u)
  .AddOptionalArg("device_memory_padding",
      R"code(**`mixed` backend only** Padding for nvJPEG's device memory allocations in bytes.
This parameter helps to avoid reallocation in nvJPEG whenever a bigger image
is encountered and internal buffer needs to be reallocated to decode it.)code",
      16*1024*1024)
  .AddOptionalArg("host_memory_padding",
      R"code(**`mixed` backend only** Padding for nvJPEG's host memory allocations in bytes.
This parameter helps to avoid reallocation in nvJPEG whenever a bigger image
is encountered and internal buffer needs to be reallocated to decode it.)code",
      8*1024*1024)  // based on ImageNet heuristics (8MB)
  .AddOptionalArg("affine",
      R"code(**`mixed` backend only** If internal threads should be affined to CPU cores)code",
      true)
  .AddOptionalArg("split_stages",
      R"code(**`mixed` backend only** Split into separated CPU stage and GPU stage operators)code",
      false)
  .AddOptionalArg("use_chunk_allocator",
      R"code(**Experimental, `mixed` backend only** Use chunk pinned memory allocator, allocating chunk of size
`batch_size*prefetch_queue_depth` during the construction and suballocate them
in runtime. Ignored when `split_stages` is false.)code",
      false)
  .AddOptionalArg("use_fast_idct",
      R"code(Enables fast IDCT in CPU based decompressor when GPU implementation cannot handle given image.
According to libjpeg-turbo documentation, decompression performance is improved by 4-14% with very little
loss in quality.)code",
      false);

DALI_SCHEMA(ImageDecoder)
  .DocStr(R"code(Decode images. For jpeg images, the implementation will be based on nvJPEG
library or libjpeg-turbo depending on the selected backend (`mixed` and `cpu` respectively).
Other image formats are decoded with OpenCV or other specific libraries (e.g. libtiff).
The Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CachedDecoderAttr");

// Fused

DALI_SCHEMA(ImageDecoderCrop)
  .DocStr(R"code(Decode images with a fixed cropping window size and variable anchor.
When possible, will make use of partial decoding (e.g. libjpeg-turbo, nvJPEG).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CropAttr");

DALI_SCHEMA(ImageDecoderRandomCrop)
  .DocStr(R"code(Decode images with a random cropping anchor/window.
When possible, will make use of partial decoding (e.g. libjpeg-turbo, nvJPEG).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("RandomCropAttr");


DALI_SCHEMA(ImageDecoderSlice)
  .DocStr(R"code(Decode images on the host with a cropping window of given size and anchor.
Inputs must be supplied as 3 separate tensors in a specific order: `data`
containing input data, `anchor` containing either normalized or absolute coordinates
(depending on the value of `normalized_anchor`) for the starting point of the
slice (x0, x1, x2, ...), and `shape` containing either normalized or absolute coordinates
(depending on the value of `normalized_shape`) for the dimensions of the slice
(s0, s1, s2, ...). Both `anchor` and `shape` coordinates must be within the interval
[0.0, 1.0] for normalized coordinates, or within the image shape for absolute
coordinates. Both `anchor` and `shape` inputs will provide as many dimensions as specified
with arguments `axis_names` or `axes`. By default `ImageDecoderSlice` operator uses normalized
coordinates and `WH` order for the slice arguments.
When possible, will make use of partial decoding (e.g. libjpeg-turbo, nvJPEG).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(3)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("SliceAttr");

}  // namespace dali
