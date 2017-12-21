// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize_permute.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(CropMirrorNormalizePermute,
    CropMirrorNormalizePermute<GPUBackend>);

OPERATOR_SCHEMA(CropMirrorNormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
