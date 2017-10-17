#include "ndll/pipeline/transformer.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(CPUTransformer, Transformer<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUTransformer, Transformer<GPUBackend>);

} // namespace ndll
