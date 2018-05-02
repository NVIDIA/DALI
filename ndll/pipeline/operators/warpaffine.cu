// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/warpaffine.h"
#include "ndll/pipeline/operators/displacement_filter_impl_gpu.cuh"

namespace ndll {

NDLL_REGISTER_OPERATOR(WarpAffine, WarpAffine<GPUBackend>, GPU);

NDLL_REGISTER_OPERATOR(Rotate, Rotate<GPUBackend>, GPU);

}  // namespace ndll

