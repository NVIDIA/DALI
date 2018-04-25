// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/new_resize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(NewResize, NewResize<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(NewResize, NewResize<GPUBackend>, GPU);
NDLL_OPERATOR_SCHEMA(NewResize)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
    .AddOptionalArg("warp_resize", "Foo", false)
    .AddArg("resize_a", "Lower bound for resize")
    .AddArg("resize_b", "Upper bound for resize")
    .AddOptionalArg("image_type", "Type of the input image", NDLL_RGB)
    .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
    .AddOptionalArg("crop", "Size of the cropped image", -1)
    .AddOptionalArg("mirror_prob", "Probability of a random horizontal or "
                    "vertical flip of the image", vector<float>{0.f, 0.f})
    .AddOptionalArg("interp_type", "Type of interpolation used", NDLL_INTERP_LINEAR);

NDLL_REGISTER_TYPE(ResizeMapping, NDLL_RESIZE_MAPPING);
NDLL_REGISTER_TYPE(PixMapping, NDLL_PIX_MAPPING);
NDLL_REGISTER_TYPE(uint32_t, NDLL_UINT32);
}  // namespace ndll

