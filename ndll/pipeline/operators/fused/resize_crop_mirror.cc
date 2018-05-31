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
  .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
  .AddOptionalArg("warp_resize", "Foo", false)
  .AddArg("resize_a", "Lower bound for resize")
  .AddArg("resize_b", "Upper bound for resize")
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
  .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
  .AddOptionalArg("warp_resize", "Foo", false)
  .AddArg("resize_a", "Lower bound for resize")
  .AddArg("resize_b", "Upper bound for resize")
  .AddArg("crop", "Size of the cropped image")
  .AddOptionalArg("crop_pos_x",
      "Horizontal position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("crop_pos_y",
      "Vertical position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("mirror", "Mask for horizontal flip", 0);

}  // namespace ndll
