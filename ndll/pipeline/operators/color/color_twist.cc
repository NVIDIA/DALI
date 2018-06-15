// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/color/color_twist.h"
#include <vector>
#include <string>

namespace ndll {

NDLL_SCHEMA(ColorTransformBase)
    .DocStr(R"code(Base Schema for color transformations operators.)code")
    .AddOptionalArg("image_type",
        R"code(`ndll.types.NDLLImageType`
        The color space of input and output image)code", NDLL_RGB);

NDLL_SCHEMA(Brightness)
    .DocStr(R"code(Changes the brightness of an image)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("brightness",
        R"code(`float` or `float tensor`
        Brightness change factor.
        Values >= 0 are accepted. For example:
          `0` - black image,
          `1` - no change
          `2` - increase brightness twice
          )code", 1.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Contrast)
    .DocStr(R"code(Changes the color contrast of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("contrast",
        R"code(`float` or `float tensor`
        Contrast change factor.
        Values >= 0 are accepted. For example:
          `0` - gray image,
          `1` - no change
          `2` - increase contrast twice
          )code", 1.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Hue)
    .DocStr(R"code(Changes the hue level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue",
        R"code(`float` or `float tensor`
        Hue change in angles.
        )code", 0.f)
    .AddParent("ColorTransformBase");

NDLL_SCHEMA(Saturation)
    .DocStr(R"code(Changes saturation level of the image.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("saturation",
        R"code(`float` or `float tensor`
        Saturation change factor.
        Values >= 0 are supported. For example:
          `0` - completely desaturated image
          `1` - no change to image's saturation
          )code", 1.f)
    .AddParent("ColorTransformBase");

}  // namespace ndll
