// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/decoder/nvjpeg_decoder.h"

namespace dali {

DALI_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);

DALI_SCHEMA(nvJPEGDecoder)
  .DocStr(R"code(Decode JPEG images using the nvJPEG library.
          Output of the decoder is on the GPU
          and uses `HWC` ordering.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(`dali.types.DALIImageType`
      The color space of output image)code",
      DALI_RGB)
  .AddOptionalArg("use_batched_decode",
      R"code(`bool`
      Use nvJPEG's batched decoding API.)code", false);

}  // namespace dali
