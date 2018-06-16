// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/displacement/warpaffine.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_gpu.cuh"

namespace dali {

DALI_REGISTER_OPERATOR(WarpAffine, WarpAffine<GPUBackend>, GPU);

}  // namespace dali
