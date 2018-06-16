// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/water.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_gpu.cuh"

namespace dali {

DALI_REGISTER_OPERATOR(Water, Water<GPUBackend>, GPU);

}  // namespace dali
