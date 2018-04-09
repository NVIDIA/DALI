// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Resize)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("random_resize", "Whether to randomly resize images")
  .AddOptionalArg("warp_resize", "Foo")
  .AddArg("resize_a", "Lower bound for resize")
  .AddArg("resize_b", "Upper bound for resize")
  .AddOptionalArg("image_type", "Output image type")
  .AddOptionalArg("interp_type", "Type of interpolation used");

}  // namespace ndll
