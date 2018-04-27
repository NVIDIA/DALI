// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/sphere.h"
#include "ndll/pipeline/operators/displacement_filter_impl_gpu.cuh"

namespace ndll {

NDLL_REGISTER_OPERATOR(Sphere, Sphere<GPUBackend>, GPU);

}  // namespace ndll
