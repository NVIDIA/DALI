// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/decoder/nvjpeg_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);

NDLL_SCHEMA(nvJPEGDecoder)
  .DocStr(R"code(Decode JPEG images using the nvJPEG library.
          Output of the decoder is on the GPU
          and uses `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(`ndll.types.NDLLImageType`
      The color space of output image)code",
      NDLL_RGB)
  .AddOptionalArg("use_batched_decode",
      R"code(`bool`
      Use nvJPEG's batched decoding API.)code", false);

}  // namespace ndll
