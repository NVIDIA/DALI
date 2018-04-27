// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dump_image.h"
#include "ndll/util/image.h"

namespace ndll {

template<>
void DumpImage<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  WriteHWCBatch(input, suffix_ + "-" + std::to_string(idx));

  // Forward the input
  output->Copy(input, ws->stream());
}

NDLL_REGISTER_OPERATOR(DumpImage, DumpImage<GPUBackend>, GPU);

}  // namespace ndll
