// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/sphere.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Sphere, Sphere<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Sphere, Sphere<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Sphere)
    .DocStr("Foo")
    .NumInput(1, INT_MAX)
    .NumOutput(1, INT_MAX);

}  // namespace ndll

