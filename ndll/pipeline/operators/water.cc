// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/water.h"
#include "ndll/pipeline/operators/displacement_filter_impl_cpu.h"

#ifndef M_PI
const float M_PI =  3.14159265358979323846;
#endif

namespace ndll {

NDLL_REGISTER_OPERATOR(Water, Water<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(Water)
    .DocStr("Perform a water augmentation")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("ampl_x", "Foo", 10.f)
    .AddOptionalArg("ampl_y", "Foo", 10.f)
    .AddOptionalArg<float>("freq_x", "Foo", 2.0 * M_PI / 128)
    .AddOptionalArg<float>("freq_y", "Foo", 2.0 * M_PI / 128)
    .AddOptionalArg("phase_x", "Foo", 0.f)
    .AddOptionalArg("phase_y", "Foo", 0.f)
    DISPLACEMENT_SCHEMA_ARGS;

}  // namespace ndll

