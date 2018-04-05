// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/ocv_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(OCVDecoder, OCVDecoder<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(OCVDecoder)
  .DocStr("Use OpenCV to decode images. More general than the JPEG-specific"
          "TJPGDecoder operator"
          "Input(0): Encoded image stream"
          "Output(0): Decoded image in HWC ordering")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll

