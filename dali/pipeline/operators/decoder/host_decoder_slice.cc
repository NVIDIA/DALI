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

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <memory>
#include "dali/error_handling.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/decoder/host_decoder_slice.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

HostDecoderSlice::HostDecoderSlice(const OpSpec &spec)
  : HostDecoder(spec)
  , SliceAttr(spec) {
}

DALI_REGISTER_OPERATOR(HostDecoderSlice, HostDecoderSlice, CPU);

DALI_SCHEMA(HostDecoderSlice)
  .DocStr(R"code(Decode images on the host with a cropping window of given size and anchor.
Inputs must be supplied as 3 tensors in a specific order: `encoded_data` containing encoded
image data, `begin` containing the starting pixel coordinates for the `crop` in `(x,y)`
format, and `size` containing the pixel dimensions of the `crop` in `(w,h)` format.
For both `begin` and `size`, coordinates must be in the interval `[0.0, 1.0]`.
When possible, will make use of partial decoding (e.g. libjpeg-turbo).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(3)
  .NumOutput(1)
  .AddParent("HostDecoder");

}  // namespace dali
