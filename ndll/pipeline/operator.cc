#include "ndll/pipeline/operator.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(CPUOperator, Operator<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUOperator, Operator<GPUBackend>);

} // namespace ndll
