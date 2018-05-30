// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/color/color_twist.h"
#include <vector>
#include <string>

namespace ndll {

NDLL_SCHEMA(ColorTransformBase)
    .DocStr("Base Schema for color transformations operators.")
    .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB);

NDLL_SCHEMA(Brightness)
    .DocStr("Changes the brightness of an image")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness", "Brightness change (0 - black image, 1 - no changes)", 1.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Contrast)
    .DocStr("Changes the color contrast of the image.")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast", "Contrast change (0 - gray image, 1 - no change)", 1.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Hue)
    .DocStr("Changes the hue level of the image.")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue", "Hue change in angles (0 - no change)", 0.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Saturation)
    .DocStr("Changes saturation level of the image.")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("saturation",  "Saturation change (1 - no change)", 1.f)
    .AddParent("ColorTransformBase");

}  // namespace ndll
