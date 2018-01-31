// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/jitter.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Jitter, Jitter<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(Jitter, Jitter<GPUBackend>);

NDLL_OPERATOR_SCHEMA(Jitter)
    .DocStr("Foo")
    .NumInput(1, INT_MAX)
    .NumOutput(1, INT_MAX);

}  // namespace ndll

