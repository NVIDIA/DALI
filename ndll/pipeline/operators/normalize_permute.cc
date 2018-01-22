// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>);

NDLL_OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .OutputFn([](const OpSpec &spec) {
      auto input_sets = spec.GetArgument<int>("num_input_sets", 1);
      NDLL_ENFORCE(spec.NumInput() % input_sets == 0);
      return spec.NumInput();
  });

}  // namespace ndll
