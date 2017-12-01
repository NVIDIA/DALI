#include "ndll/pipeline/operators/dummy_op.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(DummyOp, DummyOp<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(DummyOp, DummyOp<GPUBackend>);

} // namespace ndll
