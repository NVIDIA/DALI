#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_GPU_OPERATOR(Resize, Resize<GPUBackend>);

OPERATOR_SCHEMA(Resize)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1);

} // namespace ndll
