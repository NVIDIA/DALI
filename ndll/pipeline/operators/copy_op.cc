#include "ndll/pipeline/operators/copy_op.h"

namespace ndll {

NDLL_REGISTER_CPU_TRANSFORM(CopyOp, CopyOp<CPUBackend>);
NDLL_REGISTER_GPU_TRANSFORM(CopyOp, CopyOp<GPUBackend>);

} // namespace ndll
