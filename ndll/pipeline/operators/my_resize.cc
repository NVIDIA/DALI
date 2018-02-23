// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/my_resize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(MyResize, MyResize<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(MyResize, MyResize<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(MyResize)
    .DocStr("Foo")
    .NumInput(1, INT_MAX)
    .NumOutput(1, INT_MAX);

}  // namespace ndll

