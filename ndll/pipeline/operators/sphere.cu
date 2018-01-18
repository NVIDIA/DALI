// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/sphere.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Sphere, Sphere<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(Sphere, Sphere<GPUBackend>);

OPERATOR_SCHEMA(Sphere)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1);

__global__ void BatchedSphereKernel(const uint8 *in_batch,
                                    int H, int W, int C, uint8 *out_batch) {
        /* We process one image per thread block */
    AUGMENT_TRANSFORM_GPU(H, W, C, in_batch, out_batch, SPHERE);
}

NDLLError_t BatchedSphere(const uint8 *in_batch,
                          int N, const Dims &dims,
                          uint8 *out_batch, cudaStream_t stream) {
    BatchedSphereKernel<<<N, dim3(32, 32), 0, stream>>>
            (in_batch, dims[0], dims[1], dims[2], out_batch);
    return NDLLSuccess;
}

}  // namespace ndll
