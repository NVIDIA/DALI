// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(Resize, Resize<GPUBackend>);

OPERATOR_SCHEMA(Resize)
  .DocStr("Foo")
  .NumInput(1, INT_MAX)
  .OutputFn([](const OpSpec &spec) {
      return spec.NumInput();
  });

}  // namespace ndll
