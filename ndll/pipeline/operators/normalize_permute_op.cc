#include "ndll/pipeline/operators/normalize_permute_op.h"

namespace ndll {

NDLL_REGISTER_GPU_TRANSFORM(NormalizePermuteOp, NormalizePermuteOp<GPUBackend>);

} // namespace ndll
