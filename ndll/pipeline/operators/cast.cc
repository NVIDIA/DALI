// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/cast.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Cast, Cast<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Cast, Cast<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Cast)
  .DocStr("Cast tensor to a different type")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("dtype", "Output data type");

}  // namespace ndll
