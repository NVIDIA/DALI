// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

DALI_SCHEMA(DisplacementFilter)
  .DocStr("Base schema for displacement operators")
  .AddOptionalArg("mask",
      R"code(`int` or `int tensor`
      Whether to apply this augmentation to the input image.
        0 - do not apply this transformation
        1 - apply this transformation
        )code", 1)
  .AddOptionalArg("interp_type",
      R"code(`dali.types.DALIInterpType`
      Type of interpolation used)code",
      DALI_INTERP_NN)
  .AddOptionalArg("fill_value",
      R"code(`float`
      Color value used for padding pixels.)code",
      0.f);

}  // namespace dali
