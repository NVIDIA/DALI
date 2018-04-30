// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/sphere.h"
#include "ndll/pipeline/operators/displacement_filter_impl_cpu.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Sphere, Sphere<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(Sphere)
    .DocStr("Perform a sphere augmentation")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    DISPLACEMENT_SCHEMA_ARGS;

}  // namespace ndll


