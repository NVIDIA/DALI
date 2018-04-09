// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/tjpg_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(TJPGDecoder, TJPGDecoder<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(TJPGDecoder)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type", "Output image type");

}  // namespace ndll
