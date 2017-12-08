// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/hybrid_decode.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(HuffmanDecoder, HuffmanDecoder<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(DCTQuantInv, DCTQuantInv<GPUBackend>);

}  // namespace ndll
