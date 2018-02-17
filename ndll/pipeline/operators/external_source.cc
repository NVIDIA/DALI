// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/external_source.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(ExternalSource)
  .DocStr("Foo")
  .NumInput(0)
  .NumOutput(1);

}  // namespace ndll
