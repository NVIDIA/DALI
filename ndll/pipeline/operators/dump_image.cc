#include "ndll/pipeline/operators/dump_image.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(DumpImage, DumpImage<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(DumpImage, DumpImage<GPUBackend>);

} // namespace ndll
