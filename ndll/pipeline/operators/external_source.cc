#include "ndll/pipeline/operators/external_source.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(ExternalSource, ExternalSource<CPUBackend>);

} // namespace ndll
