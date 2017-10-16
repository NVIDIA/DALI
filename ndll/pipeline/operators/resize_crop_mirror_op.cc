#include "ndll/pipeline/operators/resize_crop_mirror_op.h"

namespace ndll {

NDLL_REGISTER_CPU_TRANSFORM(ResizeCropMirrorOp, ResizeCropMirrorOp<CPUBackend>);
NDLL_REGISTER_CPU_TRANSFORM(FastResizeCropMirrorOp, FastResizeCropMirrorOp<CPUBackend>);

} // namespace ndll
