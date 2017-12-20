// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/make_contiguous.h"

namespace ndll {
namespace internal {

NDLL_REGISTER_INTERNAL_OP(MakeContiguous, MakeContiguous);

OPERATOR_SCHEMA(MakeContiguous)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace internal
} // namespace ndll
