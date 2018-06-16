// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/sphere.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Sphere, Sphere<CPUBackend>, CPU);

DALI_SCHEMA(Sphere)
    .DocStr("Perform a sphere augmentation.")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddParent("DisplacementFilter");

}  // namespace dali


