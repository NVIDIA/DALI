// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/make_contiguous.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(MakeContiguous, MakeContiguous, Mixed);

NDLL_OPERATOR_SCHEMA(MakeContiguous)
  .DocStr("Move input batch to a contiguous representation, more suitable "
          "for execution on the GPU")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
