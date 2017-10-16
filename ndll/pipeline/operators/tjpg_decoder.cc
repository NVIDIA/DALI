#include "ndll/pipeline/operators/tjpg_decoder.h"

namespace ndll {

NDLL_REGISTER_CPU_DECODER(TJPGDecoder, TJPGDecoder<CPUBackend>);

NDLL_REGISTER_CPU_TRANSFORM(DumpImageOp, DumpImageOp<CPUBackend>);
NDLL_REGISTER_GPU_TRANSFORM(DumpImageOp, DumpImageOp<GPUBackend>);

} // namespace ndll
