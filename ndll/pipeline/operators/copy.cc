// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/copy.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Copy, Copy<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Copy, Copy<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Copy)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
