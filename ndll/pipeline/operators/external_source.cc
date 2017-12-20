// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/external_source.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(ExternalSource, ExternalSource<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(ExternalSource, ExternalSource<GPUBackend>);

OPERATOR_SCHEMA(ExternalSource)
  .DocStr("Foo")
  .NumInput(0)
  .NumOutput(1);

} // namespace ndll
