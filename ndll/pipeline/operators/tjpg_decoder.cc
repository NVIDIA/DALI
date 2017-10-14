#include "ndll/pipeline/operators/tjpg_decoder.h"

namespace ndll {

NDLL_REGISTER_CPU_DECODER(TJPGDecoder, TJPGDecoder<CPUBackend>);

} // namespace ndll
