// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize_crop_mirror.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(ResizeCropMirror)
  .DocStr("Perform a fused resize, crop, mirror operation. Handles both fixed"
          " and random resizing and cropping.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
  .AddOptionalArg("warp_resize", "Foo", false)
  .AddArg("resize_a", "Lower bound for resize")
  .AddArg("resize_b", "Upper bound for resize")
  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
  .AddArg("crop", "Size of the cropped image")
  .AddOptionalArg("mirror_prob", "Probablity of random flipping of the image", 0.5f);

NDLL_REGISTER_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(FastResizeCropMirror)
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
  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
  .AddArg("crop", "Size of the cropped image")
  .AddOptionalArg("mirror_prob", "Probablity of random flipping of the image", 0.5f);

}  // namespace ndll
