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
#include "dali/pipeline/operators/decoder/host_decoder_crop.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

HostDecoderCrop::HostDecoderCrop(const OpSpec &spec)
  : HostDecoder(spec)
  , CropAttr(spec)
  , per_sample_crop_window_generators_(batch_size_) {
  for (int i = 0; i < batch_size_; i++) {
    DALI_ENFORCE(crop_height_[i] > 0 && crop_width_[i],
      "crop window dimensions not provided");
  }
}

void HostDecoderCrop::SetupSharedSampleParams(SampleWorkspace *ws) {
  const auto data_idx = ws->data_idx();
  const auto crop_x_norm = spec_.GetArgument<float>("crop_pos_x", ws, data_idx);
  const auto crop_y_norm = spec_.GetArgument<float>("crop_pos_y", ws, data_idx);

  per_sample_crop_window_generators_[data_idx] =
    [this, data_idx, crop_x_norm, crop_y_norm](int H, int W) {
      CropWindow crop_window;
      crop_window.h = crop_height_[data_idx];
      crop_window.w = crop_width_[data_idx];
      std::tie(crop_window.y, crop_window.x) =
        CalculateCropYX(
          crop_y_norm, crop_x_norm,
          crop_window.h, crop_window.w,
          H, W);
      DALI_ENFORCE(crop_window.IsInRange(H, W));
      return crop_window;
    };
}

DALI_REGISTER_OPERATOR(HostDecoderCrop, HostDecoderCrop, CPU);

DALI_SCHEMA(HostDecoderCrop)
  .DocStr(R"code(Decode images on the host with a specified cropping anchor/window.
When possible, will make use of partial decoding (e.g. libjpeg-turbo).
When not supported, will decode the whole image and then crop.
Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("crop_pos_x",
      R"code(Horizontal position of the crop in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .AddOptionalArg("crop_pos_y",
      R"code(Vertical position of the crop in image coordinates (0.0 - 1.0))code",
      0.5f, true)
  .AddOptionalArg("crop",
      R"code(Size of the cropped image. If only a single value `c` is provided,
      the resulting crop will be square with size `(c,c)`)code",
      std::vector<float>{0.f, 0.f})
  .AddParent("HostDecoder");

}  // namespace dali
