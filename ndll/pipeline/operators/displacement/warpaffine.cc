// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/displacement/warpaffine.h"
#include "ndll/pipeline/operators/displacement/displacement_filter_impl_cpu.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(WarpAffine, WarpAffine<CPUBackend>, CPU);

NDLL_SCHEMA(WarpAffine)
    .DocStr(R"code(Apply an affine transformation to the image)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddArg("matrix",
        R"code(`list of float`
        Matrix of the transform (dst -> src).
        Given list of values `(M11, M12, M13, M21, M22, M23)`
        this operation will produce a new image using  formula
        ```
        dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)
        ```
        It is equivalent to OpenCV's `warpAffine` operation
        with a flag `WARP_INVERSE_MAP` set)code")
    .AddOptionalArg("use_image_center",
        R"code(`bool`
        Whether to use image center as the center of transformation.
        When this is `true` coordinates are calculated from the center of the image)code",
        false)
    .AddParent("DisplacementFilter");

}  // namespace ndll
