// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/decoder/host_decoder.h"

namespace dali {

DALI_REGISTER_OPERATOR(HostDecoder, HostDecoder, CPU);

DALI_SCHEMA(HostDecoder)
  .DocStr(R"code(Decode images on the host using OpenCV.
          When applicable, it will pass execution to faster,
          format-specific decoders (like libjpeg-turbo).
          Output of the decoder is in `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(`dali.types.DALIImageType`
      The color space of output image)code",
      DALI_RGB);

}  // namespace dali

