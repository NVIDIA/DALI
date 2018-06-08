// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/jitter.h"
#include "ndll/pipeline/operators/util/randomizer_impl_gpu.cuh"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_gpu.cuh"

namespace ndll {

NDLL_REGISTER_OPERATOR(Jitter, Jitter<GPUBackend>, GPU);

}  // namespace ndll

