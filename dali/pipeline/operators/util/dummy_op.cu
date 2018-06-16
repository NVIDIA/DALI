// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/dummy_op.h"

namespace dali {

DALI_REGISTER_OPERATOR(DummyOp, DummyOp<GPUBackend>, GPU);

}  // namespace dali

