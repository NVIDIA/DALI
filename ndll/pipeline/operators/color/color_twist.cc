// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/color/color_twist.h"
#include <vector>
#include <string>

namespace ndll {

NDLL_SCHEMA(ColorTransformBase)
    .DocStr(R"code(Base Schema for color transformations operators.)code")
    .AddOptionalArg("image_type",
        R"code(The color space of input and output image)code", NDLL_RGB);

NDLL_SCHEMA(Brightness)
    .DocStr(R"code(Changes the brightness of an image)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
        R"code(`float`Brightness change factor.
        Values >= 0 are accepted. For example:
          `0` - black image,
          `1` - no change
          `2` - increase brightness twice)code", 1.f)
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
