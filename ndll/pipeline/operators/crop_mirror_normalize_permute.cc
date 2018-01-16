// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize_permute.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(CropMirrorNormalizePermute,
    CropMirrorNormalizePermute<GPUBackend>);

OPERATOR_SCHEMA(CropMirrorNormalizePermute)
  .DocStr("Foo")
  .NumInput(1, INT_MAX)
  .OutputFn([](const OpSpec &spec) {
      auto input_sets = spec.GetArgument<int>("num_input_sets", 1);
      NDLL_ENFORCE(spec.NumInput() % input_sets == 0);
      return spec.NumInput();
  });

}  // namespace ndll
