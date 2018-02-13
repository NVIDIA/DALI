// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/make_contiguous.h"

namespace ndll {

NDLL_REGISTER_INTERNAL_OP(MakeContiguous, MakeContiguous);

NDLL_OPERATOR_SCHEMA(MakeContiguous)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
