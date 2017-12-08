// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize_crop_mirror.h"

namespace ndll {

NDLL_REGISTER_CPU_OPERATOR(ResizeCropMirror, ResizeCropMirror<CPUBackend>);
NDLL_REGISTER_CPU_OPERATOR(FastResizeCropMirror, FastResizeCropMirror<CPUBackend>);

}  // namespace ndll
