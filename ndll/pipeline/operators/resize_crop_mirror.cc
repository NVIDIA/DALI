// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize_crop_mirror.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(ResizeCropMirror)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets();

NDLL_REGISTER_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(FastResizeCropMirror)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets();

}  // namespace ndll
