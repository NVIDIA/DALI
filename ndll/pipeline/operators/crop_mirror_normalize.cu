// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<GPUBackend>, GPU);

}  // namespace ndll
