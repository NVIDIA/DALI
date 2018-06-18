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

#include "dali/pipeline/operators/fused/resize_crop_mirror.h"

namespace dali {

DALI_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);

DALI_SCHEMA(ResizeCropMirror)
  .DocStr("Perform a fused resize, crop, mirror operation. Handles both fixed"
          " and random resizing and cropping.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("resize_x", "The length of the X dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_y` is left at 0, then the op will keep "
      "the aspect ratio of the original image", 0.f)
  .AddOptionalArg("resize_y", "The length of the Y dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_x` is left at 0, then the op will keep "
      "the aspect ratio of the original image", 0.f)
  .AddOptionalArg("resize_shorter", "The length of the shorter dimension of the resized image. "
      "This option is mutually exclusive with `resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image", 0.f)
  .AddArg("crop", "Size of the cropped image")
  .AddOptionalArg("crop_pos_x",
      "Horizontal position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("crop_pos_y",
      "Vertical position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("mirror", "Mask for horizontal flip", 0);

DALI_REGISTER_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>, CPU);

DALI_SCHEMA(FastResizeCropMirror)
  .DocStr("Perform a fused resize, crop, mirror operation. Handles both fixed "
          "and random resizing and cropping. Backprojects the desired crop "
          "through the resize operation to reduce the amount of work performed")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("resize_x", "The length of the X dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_y` is left at 0, then the op will keep "
      "the aspect ratio of the original image", 0.f)
  .AddOptionalArg("resize_y", "The length of the Y dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_x` is left at 0, then the op will keep "
      "the aspect ratio of the original image", 0.f)
  .AddOptionalArg("resize_shorter", "The length of the shorter dimension of the resized image. "
      "This option is mutually exclusive with `resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image", 0.f)
  .AddArg("crop", "Size of the cropped image")
  .AddOptionalArg("crop_pos_x",
      "Horizontal position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("crop_pos_y",
      "Vertical position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("mirror", "Mask for horizontal flip", 0);

}  // namespace dali
