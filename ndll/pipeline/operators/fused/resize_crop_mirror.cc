// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/fused/resize_crop_mirror.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);

NDLL_SCHEMA(ResizeCropMirror)
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

NDLL_REGISTER_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>, CPU);

NDLL_SCHEMA(FastResizeCropMirror)
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

}  // namespace ndll
