// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/jitter.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Jitter, Jitter<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(Jitter)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("nDegree", "Foo", 2);

}  // namespace ndll
