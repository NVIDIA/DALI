// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype", "Output data type", NDLL_FLOAT)
  .AddArg("height", "Height of the input image")
  .AddArg("width", "Width of the input image")
  .AddArg("channels", "Number of channels of input image")
  .AddArg("mean", "Mean values of pixels for image normalizations")
  .AddArg("std", "Standard deviation for image normalization");

}  // namespace ndll
