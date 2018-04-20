// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dummy_op.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(DummyOp, DummyOp<GPUBackend>, GPU);

}  // namespace ndll

