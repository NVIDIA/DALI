#include "ndll/pipeline/operators/resize_op.h"

namespace ndll {

NDLL_REGISTER_GPU_TRANSFORM(ResizeOp, ResizeOp<GPUBackend>);

} // namespace ndll
