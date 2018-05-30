// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
NDLL_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);
NDLL_DEFINE_OPTYPE_REGISTRY(SupportOperator, OperatorBase);

}  // namespace ndll
