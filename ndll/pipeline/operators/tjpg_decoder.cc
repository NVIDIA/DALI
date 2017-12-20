// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/tjpg_decoder.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(TJPGDecoder, TJPGDecoder<CPUBackend>);

OPERATOR_SCHEMA(TJPGDecoder)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace ndll
