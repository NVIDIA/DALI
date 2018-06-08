// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/fused/crop_mirror_normalize.h"

namespace ndll {

NDLL_SCHEMA(CropMirrorNormalize)
  .DocStr("Perform fused cropping (fixed and random), normalization, format conversion "
          "(NHWC to NCHW) if desired, and type casting")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype", "Output data type", NDLL_FLOAT)
  .AddOptionalArg("output_layout", "Output data layout", NDLL_NCHW)
  .AddOptionalArg("pad_output", "Whether to pad the output to "
      "number of channels being multiple of 4",
      false)
  .AddOptionalArg("crop_pos_x",
      "Horizontal position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("crop_pos_y",
      "Vertical position of the crop in image coordinates (0.0 - 1.0)",
      0.5f)
  .AddOptionalArg("mirror", "Mask for horizontal flip", 0)
  .AddOptionalArg("image_type", "Type of the input image", NDLL_RGB)
  .AddArg("mean", "Mean pixel values for image normalization")
  .AddArg("std", "Standard deviation values for image normalization")
  .AddArg("crop", "Size of the cropped image");


}  // namespace ndll
