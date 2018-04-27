// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/host_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(HostDecoder, HostDecoder, CPU);

NDLL_OPERATOR_SCHEMA(HostDecoder)
  .DocStr("Decode images on the host."
          "Will pass to faster format-specific decoders if possible. "
          "Input(0): Encoded image stream"
          "Output(0): Decoded image in HWC ordering")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type", "Output color format", NDLL_RGB);

}  // namespace ndll

