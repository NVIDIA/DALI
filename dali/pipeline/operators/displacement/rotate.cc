// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/rotate.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace dali {

DALI_REGISTER_OPERATOR(Rotate, Rotate<CPUBackend>, CPU);

DALI_SCHEMA(Rotate)
    .DocStr("Rotate the image")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("angle",
        R"code(`float` or `float tensor`
        Rotation angle)code")
    .AddParent("DisplacementFilter");

}  // namespace dali
