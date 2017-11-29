#include "ndll/pipeline/operators/external_source.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(ExternalSource, ExternalSource<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(ExternalSource, ExternalSource<GPUBackend>);

} // namespace ndll
