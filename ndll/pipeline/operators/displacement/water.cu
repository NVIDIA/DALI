// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/water.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_gpu.cuh"

namespace ndll {

NDLL_REGISTER_OPERATOR(Water, Water<GPUBackend>, GPU);

}  // namespace ndll
