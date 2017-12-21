// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dummy_op.h"

#include <cstdlib>
#include <ctime>

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(DummyOp, DummyOp<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(DummyOp, DummyOp<GPUBackend>);

OPERATOR_SCHEMA(DummyOp)
  .DocStr("Foo")
  .OutputFn([](const OpSpec &spec) { return 2; })
  .NumInput(0, 10)
  .NumOutput(0, 10);

}  // namespace ndll
