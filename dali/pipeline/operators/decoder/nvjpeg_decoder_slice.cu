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

#include <vector>
#include "dali/pipeline/operators/decoder/nvjpeg_decoder_slice.h"

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoderSlice, nvJPEGDecoderSlice, Mixed);

DALI_SCHEMA(nvJPEGDecoderSlice)
  .DocStr(R"code(Partially decode JPEG images using the nvJPEG library, with a cropping window of given size and anchor.
  Inputs must be supplied as 3 tensors in a specific order: `encoded_data` containing encoded
  image data, `begin` containing the starting pixel coordinates for the `crop` in `(x,y)`
  format, and `size` containing the pixel dimensions of the `crop` in `(w,h)` format.
  For both `begin` and `size`, coordinates must be in the interval `[0.0, 1.0]`.
  Output of the decoder is in `HWC` ordering.)code")
  .NumInput(3)
  .NumOutput(1)
  .AddParent("nvJPEGDecoder");

}  // namespace dali
