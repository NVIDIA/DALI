// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/make_contiguous.h"

namespace dali {

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguous, Mixed);

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation,
  more suitable for execution on the GPU)code")
  .NumInput(1)
  .NumOutput(1);

}  // namespace dali
