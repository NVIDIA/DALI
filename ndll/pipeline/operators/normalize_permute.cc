// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>);

OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .OutputFn([](const OpSpec &spec) {
      auto num_loops = spec.GetArgument<int>("num_loops", 1);
      NDLL_ENFORCE(spec.NumInput() % num_loops == 0);
      return spec.NumInput();
  });

}  // namespace ndll
