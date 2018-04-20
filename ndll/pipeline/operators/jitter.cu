// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/jitter.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Jitter, Jitter<GPUBackend>, GPU);

}  // namespace ndll

