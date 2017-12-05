#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(NormalizePermute, NormalizePermute<GPUBackend>);

} // namespace ndll
