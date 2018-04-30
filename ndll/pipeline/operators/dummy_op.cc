// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dummy_op.h"

#include <cstdlib>

namespace ndll {

NDLL_REGISTER_OPERATOR(DummyOp, DummyOp<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(DummyOp)
  .DocStr("Dummy operator for testing")
  .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
  .NumInput(0, 10)
  .AddOptionalArg("num_outputs", "Number of outputs", 2);

}  // namespace ndll
