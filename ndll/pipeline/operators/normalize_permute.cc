// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>);

OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace ndll
