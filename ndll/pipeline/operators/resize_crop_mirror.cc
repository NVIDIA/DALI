// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize_crop_mirror.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>);

OPERATOR_SCHEMA(ResizeCropMirror)
  .DocStr("Foo")
  .NumInput(1, INT_MAX)
  .OutputFn([](const OpSpec &spec) {
      return spec.NumInput();
  });

NDLL_REGISTER_CPU_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>);

OPERATOR_SCHEMA(FastResizeCropMirror)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
