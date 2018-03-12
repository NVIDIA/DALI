// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/water.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Water, Water<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Water, Water<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Water)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("ampl_x", "Foo", 10)
    .AddOptionalArg("ampl_y", "Foo", 10)
    .AddOptionalArg("freq_x", "Foo", 2.0 * 3.1415 / 128)
    .AddOptionalArg("freq_y", "Foo", 2.0 * 3.1415 / 128)
    .AddOptionalArg("phase_x", "Foo", 0)
    .AddOptionalArg("phase_y", "Foo", 0);

}  // namespace ndll
