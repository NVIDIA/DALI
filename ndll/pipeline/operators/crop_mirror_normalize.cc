// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/crop_mirror_normalize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(CropMirrorNormalize,
    CropMirrorNormalize<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(CropMirrorNormalize)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype", "Output data type", NDLL_FLOAT)
  .AddOptionalArg("output_layout", "Output data layout", NDLL_NCHW)
  .AddOptionalArg("pad_output", "Whether to pad the output to "
      "number of channels being multiple of 4",
      false)
  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop",
      false)
  .AddOptionalArg("mirror_prob", "Probability of a random flip of the image", 0.5f)
  .AddOptionalArg("image_type", "Type of the input image", NDLL_RGB)
  .AddArg("mean", "Mean pixel values for image normalization")
  .AddArg("std", "Standard deviation values for image normalization")
  .AddArg("crop", "Size of the cropped image");


}  // namespace ndll
