// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/cast.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Cast, Cast<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(Cast, Cast<GPUBackend>);

NDLL_OPERATOR_SCHEMA(Cast)
  .DocStr("Cast tensor to a different type")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets();

}  // namespace ndll
