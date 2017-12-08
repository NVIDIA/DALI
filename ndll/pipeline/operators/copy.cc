#include "ndll/pipeline/operators/copy.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(Copy, Copy<CPUBackend>);
NDLL_REGISTER_GPU_OPERATOR(Copy, Copy<GPUBackend>);

OPERATOR_SCHEMA(Copy)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace ndll
