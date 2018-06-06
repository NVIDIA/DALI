// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/rotate.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Rotate, Rotate<CPUBackend>, CPU);

NDLL_SCHEMA(Rotate)
    .DocStr("Rotate the image")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddParent("DisplacementFilter")
    .AddArg("angle",
        R"code(`float` or `float tensor`
        Rotation angle)code");

}  // namespace ndll
