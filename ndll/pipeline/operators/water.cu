// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/water.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Water, Water<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Water, Water<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Water)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("ampl", "Foo")
    .AddOptionalArg("freq", "Foo")
    .AddOptionalArg("phase", "Foo");

#if 0
__constant__ WaveDescr waterOpGPU[2];

__global__ void BatchedWaterKernel(const uint8 *in_batch,
                                    int H, int W, int C, uint8 *out_batch) {
    AUGMENT_TRANSFORM_GPU(H, W, C, in_batch, out_batch, WATER, waterOpGPU);
}

NDLLError_t BatchedWater(const uint8 *in_batch, int N, const Dims &dims,
                         uint8 *out_batch, const dim3 &gridDim,
                         cudaStream_t stream, const WaveDescr *wParam) {
    // Copying the descriptor of operation into __constant__ memory
    cudaMemcpyToSymbol(waterOpGPU, wParam, sizeof(waterOpGPU));

    BatchedWaterKernel<<<N, gridDim, 0, stream>>>
            (in_batch, dims[0], dims[1], dims[2], out_batch);
    return NDLLSuccess;
}
#endif

}  // namespace ndll
