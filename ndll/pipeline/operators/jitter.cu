// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/jitter.h"
#include "ndll/pipeline/operators/randomizer_impl_gpu.cuh"
#include "ndll/pipeline/operators/displacement_filter_impl_gpu.cuh"

namespace ndll {

NDLL_REGISTER_OPERATOR(Jitter, Jitter<GPUBackend>, GPU);

}  // namespace ndll

