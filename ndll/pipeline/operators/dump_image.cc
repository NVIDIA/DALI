// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dump_image.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(DumpImage, DumpImage<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(DumpImage, DumpImage<GPUBackend>);

OPERATOR_SCHEMA(DumpImage)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace ndll
