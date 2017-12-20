// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/hybrid_decode.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(HuffmanDecoder, HuffmanDecoder<CPUBackend>);

OPERATOR_SCHEMA(HuffmanDecoder)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(2);

NDLL_REGISTER_GPU_OPERATOR(DCTQuantInv, DCTQuantInv<GPUBackend>);

OPERATOR_SCHEMA(DCTQuantInv)
  .DocStr("Foo")
  .NumInput(2)
  .NumOutput(1);

} // namespace ndll
