#include "ndll/pipeline/operators/hybrid_jpg_decoder.h"

namespace ndll {

NDLL_REGISTER_CPU_DECODER(HuffmanDecoder, HuffmanDecoder<CPUBackend>);
NDLL_REGISTER_GPU_TRANSFORM(DCTQuantInvOp, DCTQuantInvOp<GPUBackend>);

} // namespace ndll
