// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/sphere_new.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(SphereNew, SphereNew<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(SphereNew, SphereNew<GPUBackend>);

NDLL_OPERATOR_SCHEMA(SphereNew)
    .DocStr("Foo")
    .NumInput(1, INT_MAX)
    .NumOutput(1, INT_MAX);

}  // namespace ndll

