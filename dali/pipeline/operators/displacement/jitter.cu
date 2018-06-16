// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/jitter.h"
#include "dali/pipeline/operators/util/randomizer_impl_gpu.cuh"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_gpu.cuh"

namespace dali {

DALI_REGISTER_OPERATOR(Jitter, Jitter<GPUBackend>, GPU);

}  // namespace dali

