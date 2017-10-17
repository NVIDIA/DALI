#include "ndll/pipeline/decoder.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(CPUDecoder, Decoder<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUDecoder, Decoder<GPUBackend>);

} // namespace ndll
