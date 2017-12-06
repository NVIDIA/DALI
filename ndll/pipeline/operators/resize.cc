#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(Resize, Resize<GPUBackend>);

} // namespace ndll
