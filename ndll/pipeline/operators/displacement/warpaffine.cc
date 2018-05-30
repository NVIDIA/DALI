// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/warpaffine.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(WarpAffine, WarpAffine<CPUBackend>, CPU);

NDLL_SCHEMA(WarpAffine)
    .DocStr("Apply an affine transformation to the image")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("matrix", "Matrix of the transform (dst -> src)")
    .AddOptionalArg("use_image_center",
        "Whether to use image center as the center of transformation", false)
    DISPLACEMENT_SCHEMA_ARGS;

}  // namespace ndll
