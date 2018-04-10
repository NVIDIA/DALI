// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .OutputFn([](const OpSpec &spec) {
      auto input_sets = spec.GetArgument<int>("num_input_sets", 1);
      NDLL_ENFORCE(spec.NumInput() % input_sets == 0);
      return spec.NumInput();
  })
  .AddOptionalArg("output_dtype", "Output data type", NDLL_FLOAT)
  .AddArg("height", "Height of the input image")
  .AddArg("width", "Width of the input image")
  .AddArg("channels", "Number of channels of input image")
  .AddArg("mean", "Mean values of pixels for image normalizations")
  .AddArg("std", "Standard deviation for image normalization");

}  // namespace ndll
