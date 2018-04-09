// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dump_image.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(DumpImage, DumpImage<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(DumpImage, DumpImage<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(DumpImage)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("suffix", "Suffix to be added to output file names")
  .AddOptionalArg("input_layout", "Layout of input images");

}  // namespace ndll
