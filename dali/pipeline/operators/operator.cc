// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/operator.h"

namespace dali {

DALI_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(SupportOperator, OperatorBase);

}  // namespace dali
