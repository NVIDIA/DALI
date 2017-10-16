#include "ndll/pipeline/operators/crop_mirror_normalize_permute_op.h"

namespace ndll {

NDLL_REGISTER_CPU_TRANSFORM(CropMirrorNormalizePermuteOp,
    CropMirrorNormalizePermuteOp<CPUBackend>);

} // namespace ndll
