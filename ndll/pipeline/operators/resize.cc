// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(Resize, Resize<GPUBackend>);

}  // namespace ndll
