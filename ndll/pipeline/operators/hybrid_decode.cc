// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/hybrid_decode.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(HuffmanDecoder, HuffmanDecoder, CPU);

NDLL_OPERATOR_SCHEMA(HuffmanDecoder)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(2)
  .AddOptionalArg("dct_bytes_hint", "Hint for memory used to preallocate space per image",
      4 * 1048576);

NDLL_OPERATOR_SCHEMA(DCTQuantInv)
  .DocStr("Foo")
  .NumInput(2)
  .NumOutput(1)
  .AddOptionalArg("output_type", "Output image type", NDLL_RGB)
  .AddOptionalArg("bytes_per_sample_hint", "Hint for memory used to preallocate space per image",
      0);

}  // namespace ndll
