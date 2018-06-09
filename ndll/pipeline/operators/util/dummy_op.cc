// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/util/dummy_op.h"

#include <cstdlib>

namespace ndll {

NDLL_REGISTER_OPERATOR(DummyOp, DummyOp<CPUBackend>, CPU);

NDLL_SCHEMA(DummyOp)
  .DocStr("Dummy operator for testing")
  .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
  .NumInput(0, 10)
  .AddOptionalArg("num_outputs",
      R"code(`int`
      Number of outputs)code", 2);

}  // namespace ndll
