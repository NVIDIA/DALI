// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize_permute.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(CropMirrorNormalizePermute,
    CropMirrorNormalizePermute<GPUBackend>);

OPERATOR_SCHEMA(CropMirrorNormalizePermute)
  .DocStr("Foo")
  .NumInput(1, INT_MAX)
  .OutputFn({}(const OpSpec &spec) {
      auto num_loops = spec.GetArgument<int>("num_loops", 1);
      NDLL_ENFORCE(spec.NumInput() % num_loops == 0);
      return spec.NumInput();
  });

}  // namespace ndll
