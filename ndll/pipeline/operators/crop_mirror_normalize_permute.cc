#include "ndll/pipeline/operators/crop_mirror_normalize_permute.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(CropMirrorNormalizePermute,
    CropMirrorNormalizePermute<GPUBackend>);

} // namespace ndll
