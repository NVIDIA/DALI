// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/copy_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(CopyOp, CopyOp<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(CopyOp, CopyOp<GPUBackend>);

}  // namespace ndll
