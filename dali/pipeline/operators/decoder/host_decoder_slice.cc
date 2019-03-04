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
  , per_sample_crop_window_generators_(batch_size_) {
}

void HostDecoderSlice::DataDependentSetup(SampleWorkspace *ws, unsigned int idx) {
  DALI_ENFORCE(
    ws->NumInput() == 3,
    "Expected 3 inputs. Received: " + std::to_string(ws->NumInput()));

  const auto &images = ws->Input<CPUBackend>(0);
  const auto &crop_begin = ws->Input<CPUBackend>(1);
  const auto &crop_size = ws->Input<CPUBackend>(2);

  // Assumes xywh
  const auto crop_width = crop_size.data<float>()[0];
  const auto crop_height = crop_size.data<float>()[1];
  const auto crop_x = crop_begin.data<float>()[0];
  const auto crop_y = crop_begin.data<float>()[1];

  DALI_ENFORCE(crop_x + crop_width <= 1.0f,
    "crop_x[" + std::to_string(crop_x) + "] + crop_width["
    + std::to_string(crop_width) + "] must be <= 1.0f");
  DALI_ENFORCE(crop_y + crop_height <= 1.0f,
    "crop_y[" + std::to_string(crop_y) + "] + crop_height["
    + std::to_string(crop_height) + "] must be <= 1.0f");

  const auto data_idx = ws->data_idx();
  per_sample_crop_window_generators_[data_idx] =
    [crop_width, crop_height, crop_x, crop_y](int H, int W) {
      CropWindow crop_window;
      crop_window.y = crop_y * H;
      crop_window.x = crop_x * W;
      /*
       * To decrease floating point error, first calculate the bounding box of crop and then
       * calculate the width and height having left and top coordinates
       */
      crop_window.h = (crop_height + crop_y) * H - crop_window.y;
      crop_window.w = (crop_width + crop_x) * W - crop_window.x;
      DALI_ENFORCE(crop_window.IsInRange(H, W));
      return crop_window;
    };
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
