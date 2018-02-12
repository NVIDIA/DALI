// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operator.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(CPUOperator, Operator);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUOperator, Operator);

}  // namespace ndll
