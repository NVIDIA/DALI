#include "ndll/pipeline/operators/nvjpeg_decoder.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(nvJPEGDecoder, nvJPEGDecoder, Mixed);

NDLL_OPERATOR_SCHEMA(nvJPEGDecoder)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

}  // namespace ndll
