#include "ndll/pipeline/operators/crop_mirror_normalize_permute_op.h"

namespace ndll {

NDLL_REGISTER_GPU_TRANSFORM(CropMirrorNormalizePermuteOp,
    CropMirrorNormalizePermuteOp<GPUBackend>);

} // namespace ndll
