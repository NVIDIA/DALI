// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/nvjpeg_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);

NDLL_OPERATOR_SCHEMA(nvJPEGDecoder)
  .DocStr("Decode JPEG images using the nvJPEG library."
          "Input(0): Encoded image streams"
          "Output(0): Decoded images on GPU in HWC ordering")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type", "Output color format", NDLL_RGB)
  .AddOptionalArg("use_batched_decode",
                  "Use nvjpeg's batched decoding API", false);

}  // namespace ndll
