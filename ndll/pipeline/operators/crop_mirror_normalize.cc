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
  .AddOptionalArg("output_type", "Output data type")
  .AddOptionalArg("output_layout", "Output data layout")
  .AddOptionalArg("pad_output", "Whether to pad the output to number of channels being multiple of 4")
  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop")
  .AddOptionalArg("mirror_prob", "Probability of a random flip of the image")
  .AddOptionalArg("image_type", "Type of the input image")
  .AddOptionalArg("mean", "Mean pixel values for image normalization")
  .AddOptionalArg("std", "Standard deviation values for image normalization")
  .AddArg("crop", "Size of the cropped image");


}  // namespace ndll
