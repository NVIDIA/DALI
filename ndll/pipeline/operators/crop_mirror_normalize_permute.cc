// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize_permute.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(CropMirrorNormalizePermute,
    CropMirrorNormalizePermute<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(CropMirrorNormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets();

}  // namespace ndll
