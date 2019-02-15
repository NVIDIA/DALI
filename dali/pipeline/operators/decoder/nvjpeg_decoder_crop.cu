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
#include "dali/pipeline/operators/decoder/nvjpeg_decoder_crop.h"

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoderCrop, nvJPEGDecoderCrop, Mixed);

DALI_SCHEMA(nvJPEGDecoderCrop)
  .DocStr(R"code(Partially decode JPEG images using the nvJPEG library and a cropping window.
Output of the decoder is on the GPU and uses `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg(
    "crop",
    R"code(Size of the cropped image, specified as a pair `(crop_H, crop_W)`.
If only a single value `c` is provided, the resulting crop will be square
with size `(c,c)`)code",
    std::vector<float>{0.f, 0.f})
  .AddOptionalArg(
    "crop_pos_x",
    R"code(Normalized horizontal position of the crop (0.0 - 1.0).
Actual position is calculated as `crop_x = crop_x_norm * (W - crop_W)`,
where `crop_x_norm` is the normalized position, `W` is the width of the image
and `crop_W` is the width of the cropping window)code",
    0.5f, true)
  .AddOptionalArg(
    "crop_pos_y",
    R"code(Normalized vertical position of the crop (0.0 - 1.0).
Actual position is calculated as `crop_y = crop_y_norm * (H - crop_H)`,
where `crop_y_norm` is the normalized position, `H` is the height of the image
and `crop_H` is the height of the cropping window)code",
    0.5f, true)
  .AddParent("nvJPEGDecoder");

void nvJPEGDecoderCrop::SetupSharedSampleParams(MixedWorkspace *ws) {
  for (int data_idx = 0; data_idx < batch_size_; data_idx++) {
    float crop_x_norm = spec_.GetArgument<float>("crop_pos_x", ws, data_idx);
    float crop_y_norm = spec_.GetArgument<float>("crop_pos_y", ws, data_idx);

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
}

}  // namespace dali
